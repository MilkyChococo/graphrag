from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = SCRIPT_DIR / "qwen_predictions" / "docvqa_qwen_predictions.jsonl"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "qwen_predictions" / "docvqa_qwen_submission.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DocVQA prediction JSONL to JSON."
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--format",
        choices=["submission", "records"],
        default="submission",
        help="submission writes {data:[{questionId, answer}]}; records writes the full JSONL records list.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object on line {line_number}")
            records.append(payload)
    return records


def build_submission(records: list[dict[str, Any]]) -> dict[str, Any]:
    deduped: dict[int, dict[str, Any]] = {}
    for record in records:
        question_id = int(record["questionId"])
        deduped[question_id] = record
    return {
        "data": [
            {
                "questionId": question_id,
                "answer": str(record.get("answer", "")),
            }
            for question_id, record in sorted(deduped.items())
        ]
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input_jsonl)
    payload: Any = records if args.format == "records" else build_submission(records)
    write_json(args.output_json, payload)
    print(f"Wrote {len(records)} record(s) to {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
