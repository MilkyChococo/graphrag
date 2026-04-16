from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ViTextVQA OCR outputs into GraphRAG-ready JSON documents."
    )
    parser.add_argument(
        "--ocr-root",
        type=Path,
        default=Path("dev/ocr"),
        help="Root folder containing one subdirectory per image with ocr_results.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vitextvqa_ocr_dev/input/dev_ocr_documents.json"),
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of images to convert.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Optional detection score threshold for OCR rows.",
    )
    parser.add_argument(
        "--image-id",
        dest="image_ids",
        action="append",
        default=[],
        help=(
            "Specific image_id to convert. Repeat the flag or pass a comma-separated "
            "list, e.g. --image-id 10192 or --image-id 10192,10590."
        ),
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split())


def parse_image_ids(values: list[str]) -> list[str]:
    image_ids: list[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        image_ids.extend(part for part in parts if part)
    return image_ids


def load_ocr_rows(path: Path, min_score: float) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return []

    filtered: list[dict[str, Any]] = []
    for row in rows:
        text = normalize_text(row.get("text", ""))
        score = float(row.get("det_score", 0.0) or 0.0)
        if not text or score < min_score:
            continue
        filtered.append({
            **row,
            "text": text,
            "det_score": score,
        })

    return sorted(filtered, key=lambda row: (row.get("order", 0), row.get("index", 0)))


def build_document(image_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None

    first = rows[0]
    image_id = str(first.get("image_id", image_dir.name))
    image_name = str(first.get("image_name", f"{image_id}.jpg"))

    ordered_lines = [row["text"] for row in rows]
    bbox_lines = [
        f"{index}. bbox={row.get('bbox')} text={row['text']}"
        for index, row in enumerate(rows, start=1)
    ]

    text = "\n".join([
        "Dataset: ViTextVQA OCR",
        "Split: dev",
        f"Image ID: {image_id}",
        f"Image file: {image_name}",
        f"OCR line count: {len(rows)}",
        "Ordered OCR text:",
        *[f"{index}. {line}" for index, line in enumerate(ordered_lines, start=1)],
        "OCR regions with bounding boxes:",
        *bbox_lines,
        "Plain OCR transcript:",
        " | ".join(ordered_lines),
    ])

    return {
        "id": f"vitextvqa-ocr-dev-{image_id}",
        "title": f"ViTextVQA OCR dev image {image_id}",
        "text": text,
        "image_id": image_id,
        "image_name": image_name,
        "line_count": len(rows),
        "ocr_source_dir": str(image_dir.as_posix()),
        "ocr_lines": ordered_lines,
    }


def main() -> None:
    args = parse_args()
    image_ids = parse_image_ids(args.image_ids)

    if image_ids:
        image_dirs = [args.ocr_root / image_id for image_id in image_ids]
        missing = [path.name for path in image_dirs if not path.is_dir()]
        if missing:
            missing_str = ", ".join(missing)
            msg = f"Requested image_id directories not found under {args.ocr_root}: {missing_str}"
            raise FileNotFoundError(msg)
    else:
        image_dirs = sorted(path for path in args.ocr_root.iterdir() if path.is_dir())
        if args.limit is not None:
            image_dirs = image_dirs[: args.limit]

    documents: list[dict[str, Any]] = []
    for image_dir in image_dirs:
        ocr_path = image_dir / "ocr_results.json"
        if not ocr_path.exists():
            continue
        rows = load_ocr_rows(ocr_path, args.min_score)
        document = build_document(image_dir, rows)
        if document is not None:
            documents.append(document)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(documents)} OCR documents to {args.output.as_posix()}"
    )


if __name__ == "__main__":
    main()
