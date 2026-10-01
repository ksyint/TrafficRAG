"""Leave-video-out caption retrieval and violation-vote diagnostics."""

from collections import Counter, defaultdict

import numpy as np

from trafficrag.retrieval.index import CudaCosineIndex


def relevance_metrics(indices, labels, targets):
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    targets = np.asarray(targets)
    if indices.ndim != 2 or len(indices) != len(targets):
        raise ValueError("Neighbor indices and target labels must align")
    if not np.isin(labels, (0, 1)).all() or not np.isin(targets, (0, 1)).all():
        raise ValueError("Retrieval diagnostics require binary labels")
    if np.any((indices < 0) | (indices >= len(labels))):
        raise ValueError("Retrieved neighbor IDs exceed the label table")
    relevant = labels[indices] == targets[:, None]
    ranks = np.arange(1, indices.shape[1] + 1)
    first = np.argmax(relevant, axis=1) + 1
    reciprocal = np.where(relevant.any(1), 1 / first, 0)
    cumulative = relevant.cumsum(1) / ranks
    average_precision = (cumulative * relevant).sum(1) / np.maximum(1, relevant.sum(1))
    discount = 1 / np.log2(ranks + 1)
    dcg = (relevant * discount).sum(1)
    ideal = np.asarray([discount[: int(count)].sum() for count in relevant.sum(1)])
    return {
        "queries": len(targets),
        "k": indices.shape[1],
        "label_precision": float(relevant.mean()),
        "hit_rate": float(relevant.any(1).mean()),
        "mean_reciprocal_rank": float(reciprocal.mean()),
        "truncated_average_precision": float(average_precision.mean()),
        "truncated_ndcg": float(
            np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0).mean()
        ),
    }


def vote_predictions(indices, labels):
    labels = np.asarray(labels)
    neighbors = labels[np.asarray(indices)]
    if not np.isin(neighbors, (0, 1)).all():
        raise ValueError("Knowledge-base voting requires binary labels")
    return neighbors.mean(1)


def neighbor_diversity(indices, records):
    rows = []
    for neighbors in indices:
        selected = [records.rows[int(index)] for index in neighbors]
        videos = Counter(row.video_id for row in selected)
        captions = Counter(row.caption.strip().casefold() for row in selected)
        probabilities = np.asarray(list(videos.values()), dtype=float) / len(selected)
        rows.append(
            {
                "distinct_videos": len(videos),
                "distinct_groups": len({row.group for row in selected}),
                "distinct_captions": len(captions),
                "largest_video_share": max(videos.values()) / len(selected),
                "video_entropy": float(-(probabilities * np.log(probabilities)).sum()),
            }
        )
    return {
        "queries": len(rows),
        "means": {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        if rows
        else {},
        "per_query": rows,
    }


def memory_diagnostics(records, vectors, ks=(1, 3, 5, 10), device="cuda", batch_size=64):
    index = CudaCosineIndex(vectors, device)
    labels = np.asarray([row.label for row in records])
    exclusions = records.exclusions(
        video_ids=[row.video_id for row in records],
        groups=[row.group for row in records],
    )
    maximum = min(len(records) - len(values) for values in exclusions)
    if maximum < 1:
        raise ValueError("Leave-video-out retrieval requires at least two independent video groups")
    k = min(max(ks), maximum)
    result = index.search_batches(vectors, k=k, batch_size=batch_size, exclusions=exclusions)
    reports = []
    for requested in sorted(set(ks)):
        if requested < 1:
            raise ValueError("Retrieval cutoffs must be positive")
        used = min(requested, result.indices.shape[1])
        neighbors = result.indices[:, :used]
        scores = vote_predictions(neighbors, labels)
        report = relevance_metrics(neighbors, labels, labels)
        report.update(
            requested_k=requested,
            violation_vote_accuracy=float(((scores >= 0.5) == labels).mean()),
            mean_positive_vote=float(scores[labels == 1].mean()) if (labels == 1).any() else None,
            mean_negative_vote=float(scores[labels == 0].mean()) if (labels == 0).any() else None,
            diversity=neighbor_diversity(neighbors, records),
        )
        reports.append(report)
    difficult = []
    votes = vote_predictions(result.indices, labels)
    for query, (row, vote) in enumerate(zip(records, votes)):
        if int(vote >= 0.5) == row.label:
            continue
        difficult.append(
            {
                "id": row.identifier,
                "label": row.label,
                "violation_vote": float(vote),
                "neighbors": [
                    {
                        "id": records.rows[int(neighbor)].identifier,
                        "label": records.rows[int(neighbor)].label,
                        "similarity": float(similarity),
                    }
                    for neighbor, similarity in zip(
                        result.indices[query], result.similarities[query]
                    )
                ],
            }
        )
    return {
        "summary": records.summary(),
        "cutoffs": reports,
        "difficult_queries": difficult,
        "index": index.describe(),
    }


def caption_conflicts(records):
    groups = defaultdict(list)
    for row in records:
        groups[" ".join(row.caption.casefold().split())].append(row)
    return [
        {
            "caption": caption,
            "ids": [row.identifier for row in rows],
            "labels": [row.label for row in rows],
        }
        for caption, rows in groups.items()
        if len({row.label for row in rows}) > 1
    ]


def evaluate_queries(
    memory, vectors, queries, query_vectors, ks=(1, 3, 5, 10), device="cuda", batch_size=64
):
    from trafficrag.evaluation.classification import classification_summary
    from trafficrag.retrieval.records import memory_overlap

    if memory.domain != queries.domain:
        raise ValueError("Retrieval queries and memory must use the same traffic domain")
    cutoffs = sorted(set(int(value) for value in ks))
    if not cutoffs or min(cutoffs) < 1:
        raise ValueError("Retrieval evaluation requires positive cutoffs")
    values = np.asarray(query_vectors)
    if values.ndim != 2 or len(values) != len(queries):
        raise ValueError("Query embeddings must align with the caption records")
    overlap = memory_overlap(memory, queries)
    if overlap["conflicting_ids"]:
        raise ValueError("Query IDs conflict with knowledge-base records")
    exclusions = memory.exclusions(
        video_ids=[row.video_id for row in queries],
        groups=[row.group for row in queries],
    )
    for position, row in enumerate(queries):
        if row.identifier in memory.by_id:
            exclusions[position] = sorted(
                set(exclusions[position]) | {memory.by_id[row.identifier]}
            )
    index = CudaCosineIndex(vectors, device)
    result = index.search_batches(values, max(cutoffs), batch_size, exclusions)
    labels = np.asarray([row.label for row in memory])
    targets = np.asarray([row.label for row in queries])
    reports = []
    for requested in cutoffs:
        used = min(requested, result.indices.shape[1])
        neighbors = result.indices[:, :used]
        votes = vote_predictions(neighbors, labels)
        reports.append(
            {
                "requested_k": requested,
                "used_k": used,
                "ranking": relevance_metrics(neighbors, labels, targets),
                "classification": classification_summary(votes, targets),
                "diversity": neighbor_diversity(neighbors, memory),
            }
        )
    details = []
    hard_negatives = []
    votes = vote_predictions(result.indices, labels)
    for position, row in enumerate(queries):
        neighbors = []
        for number, (neighbor, similarity) in enumerate(
            zip(result.indices[position], result.similarities[position]), 1
        ):
            candidate = memory.rows[int(neighbor)]
            neighbors.append(
                {
                    "rank": number,
                    "id": candidate.identifier,
                    "video_id": candidate.video_id,
                    "group": candidate.group,
                    "label": candidate.label,
                    "caption": candidate.caption,
                    "similarity": float(similarity),
                }
            )
        positives = [entry for entry in neighbors if entry["label"] == row.label]
        negatives = [entry for entry in neighbors if entry["label"] != row.label]
        margin = (
            positives[0]["similarity"] - negatives[0]["similarity"]
            if positives and negatives
            else None
        )
        details.append(
            {
                "id": row.identifier,
                "video_id": row.video_id,
                "group": row.group,
                "label": row.label,
                "prediction": int(votes[position] >= 0.5),
                "positive_vote": float(votes[position]),
                "excluded_entries": len(exclusions[position]),
                "nearest_label_margin": margin,
                "neighbors": neighbors,
            }
        )
        if negatives:
            hard_negatives.append(
                {
                    "query_id": row.identifier,
                    "query_caption": row.caption,
                    "query_label": row.label,
                    "negative": negatives[0],
                    "positive": positives[0] if positives else None,
                    "margin": margin,
                }
            )
    return {
        "domain": memory.domain,
        "memory": memory.summary(),
        "queries": queries.summary(),
        "overlap": overlap,
        "exclusion_policy": "same video, recording group, or segment ID",
        "cutoffs": reports,
        "predictions": details,
        "hard_negatives": hard_negatives,
        "index": index.describe(),
    }
