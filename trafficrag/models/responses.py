"""Grounding response parsing with explicit crop-relative timestamp validation."""

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingResponse:
    violation: bool
    interval: object
    raw: str

    def as_dict(self):
        return {
            "violation": self.violation,
            "interval": list(self.interval) if self.interval is not None else None,
            "raw": self.raw,
        }


def json_objects(text):
    decoder = json.JSONDecoder()
    offset = 0
    objects = []
    while offset < len(text):
        start = text.find("{", offset)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        offset = start + consumed
    return objects


def timestamp_value(value):
    if isinstance(value, bool):
        raise ValueError("Boolean values are not timestamps")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        parts = value.strip().split(":")
        if len(parts) == 1:
            result = float(parts[0])
        elif len(parts) in (2, 3):
            values = [float(part) for part in parts]
            if any(number < 0 for number in values) or any(number >= 60 for number in values[1:]):
                raise ValueError("Clock-form timestamps must use valid minute and second fields")
            result = 0.0
            for number in values:
                result = result * 60 + number
        else:
            raise ValueError("Unrecognized timestamp format")
    else:
        raise ValueError("Timestamp values must be numbers or clock strings")
    if not math.isfinite(result):
        raise ValueError("Grounding timestamps must be finite")
    return result


def parse_grounding(text, duration, tolerance=1e-6):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Grounding response is empty")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Grounding crop duration must be finite and positive")
    candidates = [
        value
        for value in json_objects(text)
        if "violation" in value or "start" in value or "end" in value
    ]
    if len(candidates) != 1:
        raise ValueError("Grounding must return exactly one event JSON object")
    value = candidates[0]
    if "violation" in value and not isinstance(value["violation"], bool):
        raise ValueError("The violation field must be a JSON Boolean")
    if value.get("violation") is False:
        if value.get("start") is not None or value.get("end") is not None:
            raise ValueError("A no-violation response cannot include temporal boundaries")
        return GroundingResponse(False, None, text)
    if "start" not in value or "end" not in value:
        raise ValueError("A positive grounding response requires start and end")
    start, end = timestamp_value(value["start"]), timestamp_value(value["end"])
    if start < 0 or start >= end or end > duration + tolerance:
        raise ValueError("Grounding timestamps must lie inside the supplied video crop")
    return GroundingResponse(True, (start, min(end, duration)), text)


def absolute_interval(response, crop_start, video_duration):
    if not response.violation:
        return None
    start, end = response.interval
    if crop_start < 0 or video_duration <= 0:
        raise ValueError("Invalid crop origin or video duration")
    result = (crop_start + start, crop_start + end)
    if result[1] > video_duration + 1e-6:
        raise ValueError("Grounded interval exceeds the source video")
    return result[0], min(result[1], video_duration)


def response_audit(rows):
    valid, errors = [], []
    for index, row in enumerate(rows):
        try:
            response = parse_grounding(row["response"], float(row["duration"]))
            valid.append({"id": row.get("id", str(index)), **response.as_dict()})
        except (ValueError, TypeError, KeyError) as error:
            errors.append(
                {
                    "id": row.get("id", str(index)),
                    "error": str(error),
                    "response": row.get("response"),
                }
            )
    return {
        "records": len(rows),
        "valid": valid,
        "errors": errors,
        "valid_fraction": len(valid) / len(rows) if rows else None,
    }
