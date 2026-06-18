from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from model_api import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_ENV_FILE,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    build_completion_model,
    complete_with_image,
    complete_text,
    extract_json_object,
    resolve_api_key,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_FILE = REPO_ROOT / "test_v1.0.json"
DEFAULT_OCR_ROOT = REPO_ROOT / "spdocvqa_ocr"
DEFAULT_IMAGES_ROOT = REPO_ROOT
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "semantic_graphs"

SYSTEM_PROMPT = """You build a compact semantic graph from OCR text for DocVQA.
Return valid JSON only. Do not use markdown.
The graph must preserve exact visible strings from OCR when possible."""

USER_PROMPT_TEMPLATE = """Build a semantic graph for one scanned document page.

Return this JSON shape:
{{
  "entities": [
    {{"title": "exact entity/value", "type": "organization|person|location|date|number|money|field|document|other", "description": "short English description", "text_unit_ids": ["..."]}}
  ],
  "relationships": [
    {{"source": "entity title", "target": "entity title", "description": "short English relation", "weight": 1.0, "text_unit_ids": ["..."]}}
  ]
}}

Rules:
- Use English descriptions.
- Keep titles exactly as they appear in OCR when possible.
- Include useful form fields, labels, values, dates, amounts, names, organizations, table headings, chart labels, and document titles.
- Prefer fewer high-value entities over noisy tokens.
- Every relationship source/target must match an entity title.
- Use only these text_unit_ids: {text_unit_ids}

OCR text:
{ocr_text}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SP-DocVQA semantic graphs from OCR for images in test_v1.0.json."
    )
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--image-detail", choices=["auto", "low", "high"], default="high")
    parser.add_argument("--max-text-unit-chars", type=int, default=6000)
    parser.add_argument("--image-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def expand_image_ids(values: list[str]) -> list[str]:
    image_ids: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                image_ids.append(item)
    return list(dict.fromkeys(image_ids))


def image_stem(image_value: str) -> str:
    return Path(str(image_value)).stem


def load_test_rows(test_file: Path) -> list[dict[str, Any]]:
    payload = load_json(test_file)
    rows = payload.get("data", [])
    return rows if isinstance(rows, list) else []


def collect_test_image_ids(test_file: Path, requested: list[str], limit: int | None) -> list[str]:
    if requested:
        image_ids = requested
    else:
        image_ids = sorted({image_stem(row["image"]) for row in load_test_rows(test_file) if row.get("image")})
    if limit is not None:
        image_ids = image_ids[:limit]
    return image_ids


def build_image_path_map(test_file: Path, images_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for row in load_test_rows(test_file):
        image_value = row.get("image")
        if not image_value:
            continue
        paths[image_stem(str(image_value))] = (images_root / str(image_value)).resolve()
    return paths


def polygon_to_bbox(points: list[Any]) -> list[float] | None:
    coords: list[tuple[float, float]] = []
    for idx in range(0, len(points), 2):
        try:
            coords.append((float(points[idx]), float(points[idx + 1])))
        except (IndexError, TypeError, ValueError):
            continue
    if not coords:
        return None
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def iter_ocr_lines(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pages = payload.get("recognitionResults")
    if not isinstance(pages, list):
        pages = payload.get("analyzeResult", {}).get("readResults", [])
    rows: list[dict[str, Any]] = []
    order = 0
    for page in pages if isinstance(pages, list) else []:
        for line in page.get("lines", []):
            text = " ".join(str(line.get("text", "")).split())
            if not text:
                continue
            order += 1
            rows.append(
                {
                    "order": order,
                    "text": text,
                    "bbox": polygon_to_bbox(line.get("boundingBox") or []),
                }
            )
    return rows


def build_text_units(image_id: str, rows: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    current: list[str] = []
    current_orders: list[int] = []

    def flush() -> None:
        if not current:
            return
        idx = len(units) + 1
        unit_id = f"{image_id}_chunk_{idx:03d}"
        units.append(
            {
                "id": unit_id,
                "line_count": len(current),
                "line_orders": current_orders.copy(),
                "text": f"Image ID: {image_id}\nOCR Transcript:\n" + "\n".join(current),
            }
        )
        current.clear()
        current_orders.clear()

    for row in rows:
        line = f"[{row['order']}] {row['text']}"
        if current and len("\n".join(current + [line])) > max_chars:
            flush()
        current.append(line)
        current_orders.append(int(row["order"]))
    flush()
    return units


def normalize_title(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_graph_payload(payload: dict[str, Any], text_unit_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities: list[dict[str, Any]] = []
    seen_entities: set[tuple[str, str]] = set()
    for item in payload.get("entities", []):
        if not isinstance(item, dict):
            continue
        title = normalize_title(item.get("title"))
        if not title:
            continue
        entity_type = normalize_title(item.get("type") or "other").lower()
        key = (title, entity_type)
        if key in seen_entities:
            continue
        seen_entities.add(key)
        ids = [str(x) for x in item.get("text_unit_ids", []) if str(x) in text_unit_ids]
        entities.append(
            {
                "title": title,
                "type": entity_type,
                "description": normalize_title(item.get("description")),
                "descriptions_raw": [normalize_title(item.get("description"))],
                "text_unit_ids": ids or sorted(text_unit_ids)[:1],
                "frequency": 1,
            }
        )

    entity_titles = {item["title"] for item in entities}
    relationships: list[dict[str, Any]] = []
    seen_relationships: set[tuple[str, str, str]] = set()
    for item in payload.get("relationships", []):
        if not isinstance(item, dict):
            continue
        source = normalize_title(item.get("source"))
        target = normalize_title(item.get("target"))
        description = normalize_title(item.get("description"))
        if source not in entity_titles or target not in entity_titles or source == target:
            continue
        key = (source, target, description)
        if key in seen_relationships:
            continue
        seen_relationships.add(key)
        ids = [str(x) for x in item.get("text_unit_ids", []) if str(x) in text_unit_ids]
        relationships.append(
            {
                "source": source,
                "target": target,
                "description": description,
                "descriptions_raw": [description],
                "text_unit_ids": ids or sorted(text_unit_ids)[:1],
                "weight": float(item.get("weight") or 1.0),
            }
        )
    return entities, relationships


def build_fallback_entities(
    rows: list[dict[str, Any]],
    text_units: list[dict[str, Any]],
    max_entities: int = 40,
) -> list[dict[str, Any]]:
    text_unit_id = str(text_units[0]["id"]) if text_units else ""
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        text = normalize_title(row.get("text"))
        if not text or text in seen:
            continue
        word_count = len(text.split())
        has_digit = any(char.isdigit() for char in text)
        looks_useful = word_count <= 12 or has_digit or ":" in text
        if not looks_useful:
            continue
        seen.add(text)
        entity_type = "number" if has_digit and word_count <= 4 else "field" if ":" in text else "other"
        entities.append(
            {
                "title": text,
                "type": entity_type,
                "description": f"OCR line {row.get('order')}: {text}",
                "descriptions_raw": [f"OCR line {row.get('order')}: {text}"],
                "text_unit_ids": [text_unit_id] if text_unit_id else [],
                "frequency": 1,
            }
        )
        if len(entities) >= max_entities:
            break
    return entities


async def build_one(
    image_id: str,
    args: argparse.Namespace,
    model: Any,
    image_path_map: dict[str, Path],
) -> dict[str, Any] | None:
    ocr_file = args.ocr_root / f"{image_id}.json"
    if not ocr_file.exists():
        return None
    rows = iter_ocr_lines(load_json(ocr_file))
    if not rows:
        return None
    text_units = build_text_units(image_id, rows, args.max_text_unit_chars)
    text_unit_ids = {unit["id"] for unit in text_units}
    prompt = USER_PROMPT_TEMPLATE.format(
        text_unit_ids=", ".join(unit["id"] for unit in text_units),
        ocr_text="\n\n".join(unit["text"] for unit in text_units),
    )
    image_path = image_path_map.get(image_id)
    image_used = bool(image_path and image_path.exists())
    if image_path and image_path.exists():
        raw = await complete_with_image(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            image_path=image_path,
            image_detail=args.image_detail,
            temperature=args.temperature,
        )
    else:
        raw = await complete_text(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=args.temperature,
        )
    parsed_payload = extract_json_object(raw)
    entities, relationships = normalize_graph_payload(parsed_payload, text_unit_ids)
    parse_status = "model_json"
    if not parsed_payload:
        parse_status = "fallback_ocr_entities"
        entities = build_fallback_entities(rows, text_units)
        relationships = []
    graph = {
        "graph_type": "spdocvqa_semantic_ocr_graph",
        "created_at": now_iso(),
        "image_id": image_id,
        "image_name": f"{image_id}.png",
        "model": {
            "provider": args.provider,
            "model": args.model,
            "api_base": args.api_base,
        },
        "source_files": {"ocr_file": str(ocr_file.resolve())},
        "image_path": str(image_path) if image_path else None,
        "image_used": image_used,
        "ocr_overview": {
            "line_count": len(rows),
            "transcript_preview": [f"[{row['order']}] {row['text']}" for row in rows[:30]],
        },
        "text_units": text_units,
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "entities": entities,
        "relationships": relationships,
        "semantic_parse_status": parse_status,
        "raw_model_output": raw,
    }
    summary = {
        "created_at": graph["created_at"],
        "image_id": image_id,
        "text_unit_count": len(text_units),
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "entity_type_counts": dict(sorted(Counter(item["type"] for item in entities).items())),
    }
    output_dir = args.output_root / image_id
    write_json(output_dir / "graph.json", graph)
    write_json(output_dir / "summary.json", summary)
    return summary


async def main_async() -> None:
    args = parse_args()
    image_ids = collect_test_image_ids(args.test_file, expand_image_ids(args.image_id), args.limit)
    image_path_map = build_image_path_map(args.test_file, args.images_root.resolve())
    api_key = resolve_api_key(args.env_file, args.api_key_env)
    model = build_completion_model(
        provider=args.provider,
        model_name=args.model,
        api_key=api_key,
        api_base=args.api_base,
        max_retries=args.max_retries,
    )

    summaries: list[dict[str, Any]] = []
    skipped_existing: list[str] = []
    missing_or_empty: list[str] = []
    failures: dict[str, str] = {}
    for image_id in tqdm(image_ids, desc="Semantic graphs", unit="image"):
        if args.resume and (args.output_root / image_id / "graph.json").exists():
            skipped_existing.append(image_id)
            continue
        try:
            summary = await build_one(image_id, args, model, image_path_map)
        except Exception as exc:
            failures[image_id] = f"{type(exc).__name__}: {exc}"
            continue
        if summary is None:
            missing_or_empty.append(image_id)
        else:
            summaries.append(summary)
        if args.sleep_seconds > 0:
            await asyncio.sleep(args.sleep_seconds)

    write_json(
        args.output_root / "build_summary.json",
        {
            "created_at": now_iso(),
            "test_file": str(args.test_file.resolve()),
            "ocr_root": str(args.ocr_root.resolve()),
            "images_root": str(args.images_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "model": args.model,
            "provider": args.provider,
            "requested_image_count": len(image_ids),
            "processed_count": len(summaries),
            "skipped_existing_count": len(skipped_existing),
            "missing_or_empty_count": len(missing_or_empty),
            "missing_or_empty_image_ids": missing_or_empty,
            "failed_count": len(failures),
            "failures": failures,
            "summaries": summaries,
        },
    )
    print(
        f"Prepared semantic graph build script output at {args.output_root.resolve()} "
        f"(processed={len(summaries)}, skipped={len(skipped_existing)}, failed={len(failures)})"
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
