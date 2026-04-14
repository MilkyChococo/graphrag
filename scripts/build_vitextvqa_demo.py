from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small GraphRAG-friendly demo corpus from ViTextVQA."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("ViTextVQA_images"),
        help="Directory containing the ViTextVQA JSON files and st_images folder.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test_gt"],
        default="dev",
        help="Dataset split to convert. test is excluded because it contains placeholder answers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Maximum number of image-level documents to emit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vitextvqa_demo/input/vitextvqa_dev_subset.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split())


def dataset_file_name(split: str) -> str:
    mapping = {
        "train": "ViTextVQA_train.json",
        "dev": "ViTextVQA_dev.json",
        "test_gt": "ViTextVQA_test_gt.json",
    }
    return mapping[split]


def build_records(dataset_root: Path, split: str, limit: int) -> list[dict[str, Any]]:
    dataset_path = dataset_root / dataset_file_name(split)
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))

    images: list[dict[str, Any]] = raw["images"]
    annotations: list[dict[str, Any]] = raw["annotations"]

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    records: list[dict[str, Any]] = []
    for image in images:
        image_id = int(image["id"])
        image_annotations = annotations_by_image.get(image_id, [])
        if not image_annotations:
            continue

        filename = str(image["filename"])
        lines = [
            "Dataset: ViTextVQA",
            f"Split: {split}",
            f"Image file: {filename}",
            "Annotated Vietnamese question and answer pairs about text visible in the image:",
        ]

        for index, annotation in enumerate(image_annotations, start=1):
            question = normalize_text(annotation["question"])
            answers = ordered_unique(
                [
                    normalize_text(answer)
                    for answer in annotation.get("answers", [])
                    if normalize_text(answer)
                ]
            )
            answer_text = " | ".join(answers) if answers else "(missing)"
            lines.append(f"{index}. Question: {question}")
            lines.append(f"   Answers: {answer_text}")

        records.append(
            {
                "id": f"vitextvqa-{split}-{image_id}",
                "title": f"ViTextVQA {split} image {image_id}",
                "text": "\n".join(lines),
                "filename": filename,
                "split": split,
                "image_id": image_id,
                "qa_count": len(image_annotations),
            }
        )

        if len(records) >= limit:
            break

    return records


def main() -> None:
    args = parse_args()
    records = build_records(args.dataset_root, args.split, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(records)} records from split={args.split} to {args.output.as_posix()}"
    )


if __name__ == "__main__":
    main()
