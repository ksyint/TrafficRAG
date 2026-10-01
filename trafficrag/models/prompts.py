"""Domain evidence prompts and bounded retrieval examples for video language stages."""

import json
from dataclasses import dataclass


DOMAIN_RULES = {
    "red-light": "The ego vehicle crosses the stop line while the relevant traffic signal is red.",
    "blind-spot-left": "A pedestrian or road user approaches on a collision trajectory from the left side.",
    "blind-spot-right": "A pedestrian or road user approaches on a collision trajectory from the right side.",
}

OBSERVATIONS = {
    "red-light": (
        "traffic light state",
        "lane alignment",
        "stop-line position",
        "ego movement",
        "crossing order",
    ),
    "blind-spot-left": (
        "left-side road user",
        "relative direction",
        "ego movement",
        "visibility",
        "collision trajectory",
    ),
    "blind-spot-right": (
        "right-side road user",
        "relative direction",
        "ego movement",
        "visibility",
        "collision trajectory",
    ),
}


@dataclass(frozen=True)
class PromptBudget:
    maximum_examples: int = 10
    example_characters: int = 600
    caption_characters: int = 1600

    def __post_init__(self):
        if self.maximum_examples < 0 or min(self.example_characters, self.caption_characters) < 1:
            raise ValueError(
                "Prompt budgets must use nonnegative example counts and positive text lengths"
            )


def domain_rule(domain):
    if domain not in DOMAIN_RULES:
        raise ValueError(f"Unknown traffic event domain: {domain}")
    return DOMAIN_RULES[domain]


def compact_text(text, maximum):
    if not isinstance(text, str):
        raise ValueError("Prompt evidence must be text")
    text = " ".join(text.split())
    if len(text) <= maximum:
        return text
    shortened = text[:maximum]
    boundary = shortened.rfind(" ")
    return shortened[:boundary] if boundary > maximum // 2 else shortened


def caption_prompt(domain):
    rule = domain_rule(domain)
    details = ", ".join(OBSERVATIONS[domain])
    return (
        "Describe the visible driving event in this video crop. "
        f"Attend to {details}. "
        f"The event criterion is: {rule} "
        "Distinguish visible observations from events that are outside the camera view. "
        "Describe the chronological order of relevant actions. "
        "Return one concise English paragraph containing visual evidence."
    )


def retrieval_examples(neighbors, budget=None):
    budget = budget or PromptBudget()
    result = []
    for row in (neighbors or [])[: budget.maximum_examples]:
        if row.get("label") not in (0, 1):
            raise ValueError("Retrieved examples must use binary labels")
        result.append(
            {
                "caption": compact_text(row.get("caption", ""), budget.example_characters),
                "label": int(row["label"]),
            }
        )
    return result


def grounding_prompt(domain, duration, caption, neighbors=None, budget=None):
    if duration <= 0:
        raise ValueError("Grounding crop duration must be positive")
    budget = budget or PromptBudget()
    examples = retrieval_examples(neighbors, budget)
    references = json.dumps(examples, ensure_ascii=False)
    return (
        f"Localize this traffic event: {domain_rule(domain)} "
        f"The video crop lasts {duration:.3f} seconds. Its first instant is timestamp zero. "
        f"Candidate observations: {compact_text(caption, budget.caption_characters)}\n"
        f"Retrieved annotated examples: {references}\n"
        "Label 1 denotes a violation and label 0 denotes a safe event. "
        "Use the current video to establish the temporal boundaries. "
        'Return only JSON {"start": number, "end": number} in seconds relative to this crop. '
        "Start at the first visible violation instant and end when the violation stops. "
        f"The timestamps must satisfy 0 <= start < end <= {duration:.3f}. "
        'If the criterion is not met, return {"violation": false}.'
    )


def prompt_identity(domain, budget=None):
    from dataclasses import asdict

    budget = budget or PromptBudget()
    return {
        "version": "traffic-evidence-prompts-v1",
        "domain": domain,
        "rule": domain_rule(domain),
        "observations": list(OBSERVATIONS[domain]),
        "budget": asdict(budget),
    }
