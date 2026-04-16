from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
from graphrag.index.operations.summarize_descriptions.description_summary_extractor import (
    SummarizeExtractor,
)
from graphrag.prompts.index.extract_graph import (
    GRAPH_EXTRACTION_PROMPT as GRAPH_EXTRACTION_PROMPT_BASE,
)
from graphrag.prompts.index.summarize_descriptions import (
    SUMMARIZE_PROMPT as SUMMARIZE_PROMPT_BASE,
)
from graphrag_llm.completion import create_completion
from graphrag_llm.config.model_config import ModelConfig
from graphrag_llm.config.types import AuthMethod
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OCR_ROOT = SCRIPT_DIR / "OCR_img"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "semantic_graphs"
DEFAULT_ENV_FILE = SCRIPT_DIR.parent / ".env"
DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_ENTITY_TYPES = [
    "organization",
    "person",
    "geo",
    "event",
    "product",
    "service",
    "facility",
]

GRAPH_EXTRACTION_PROMPT_EN_VI_OUTPUT = GRAPH_EXTRACTION_PROMPT_BASE.replace(
    "3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.",
    "3. Return output as a single list of all the entities and relationships identified in steps 1 and 2. Write entity_description and relationship_description in Vietnamese. Use **##** as the list delimiter.",
).replace(
    'Output:"""',
    "Important: descriptions must be written in Vietnamese, but the tuple format, delimiters, and completion token must stay exactly the same as shown above.\nOutput:\"\"\"",
)

SUMMARIZE_PROMPT_EN_VI_OUTPUT = (
    SUMMARIZE_PROMPT_BASE.rstrip()
    + "\nWrite the final merged description in Vietnamese.\n"
)


@dataclass(frozen=True)
class OcrLine:
    bbox_id: str
    image_id: str
    image_name: str
    order: int
    det_score: float
    text: str
    text_raw: str
    bbox: list[float] | None


@dataclass(frozen=True)
class TextUnit:
    id: str
    text: str
    line_count: int
    line_orders: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a GraphRAG-style semantic graph from OCR using Gemini."
    )
    parser.add_argument(
        "--ocr-root",
        type=Path,
        default=DEFAULT_OCR_ROOT,
        help=f"OCR root folder. Default: {DEFAULT_OCR_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root folder. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=f".env file used to resolve GEMINI_API_KEY. Default: {DEFAULT_ENV_FILE}",
    )
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable holding the Gemini API key.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for Gemini calls. Default: 3",
    )
    parser.add_argument(
        "--image-id",
        action="append",
        default=[],
        help="Image id to process. Repeat the flag or pass comma-separated ids.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every OCR folder under --ocr-root. Use carefully because it can be expensive.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of images to process after filtering.",
    )
    parser.add_argument(
        "--entity-types",
        default=",".join(DEFAULT_ENTITY_TYPES),
        help="Comma-separated entity types passed to the GraphRAG extraction prompt.",
    )
    parser.add_argument(
        "--max-gleanings",
        type=int,
        default=1,
        help="Number of follow-up extraction passes, same idea as GraphRAG max_gleanings.",
    )
    parser.add_argument(
        "--max-text-unit-tokens",
        type=int,
        default=1400,
        help="Maximum approximate tokens per OCR text unit before splitting.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Discard OCR rows whose det_score is below this threshold.",
    )
    parser.add_argument(
        "--summary-max-length",
        type=int,
        default=220,
        help="Target word limit used by the description summarization prompt.",
    )
    parser.add_argument(
        "--summary-max-input-tokens",
        type=int,
        default=3000,
        help="Token budget used when summarizing merged descriptions.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the post-merge summarization step and keep raw description lists joined together.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and settings without calling Gemini.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that already have semantic graph outputs in --output-root.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Delay between images to reduce rate-limit pressure. Default: 2.0",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def expand_image_ids(values: list[str]) -> list[str]:
    image_ids: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                image_ids.append(item)
    return image_ids


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def resolve_api_key(args: argparse.Namespace) -> str:
    env_value = os.environ.get(args.api_key_env)
    if env_value:
        return env_value

    file_values = parse_env_file(args.env_file)
    env_value = file_values.get(args.api_key_env)
    if env_value:
        os.environ[args.api_key_env] = env_value
        return env_value

    raise SystemExit(
        f"Missing API key. Expected {args.api_key_env} in environment or {args.env_file}"
    )


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def mojibake_score(text: str) -> int:
    return sum(text.count(token) for token in ("Ã", "Æ", "Ð", "�"))


def maybe_fix_mojibake(text: str) -> str:
    if not text:
        return ""
    repaired = text
    for source_encoding, target_encoding in (("latin1", "utf-8"), ("cp1252", "utf-8")):
        try:
            candidate = text.encode(source_encoding).decode(target_encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if mojibake_score(candidate) < mojibake_score(repaired):
            repaired = candidate
    return repaired


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_ocr_file(image_dir: Path) -> Path | None:
    for filename in ("ocr_results.json", "ocr_results.jsonl"):
        candidate = image_dir / filename
        if candidate.exists():
            return candidate
    return None


def load_ocr_rows(image_dir: Path) -> list[dict[str, Any]]:
    ocr_file = find_ocr_file(image_dir)
    if ocr_file is None:
        return []
    if ocr_file.suffix == ".jsonl":
        return load_jsonl(ocr_file)
    payload = load_json(ocr_file)
    return payload if isinstance(payload, list) else []


def parse_bbox(row: dict[str, Any]) -> list[float] | None:
    bbox = row.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(value) for value in bbox]
    return None


def rows_to_lines(rows: list[dict[str, Any]], min_score: float) -> list[OcrLine]:
    lines: list[OcrLine] = []
    for index, row in enumerate(rows, start=1):
        det_score = float(row.get("det_score", 0.0) or 0.0)
        if det_score < min_score:
            continue

        raw_text = str(row.get("text") or "")
        text = normalize_whitespace(maybe_fix_mojibake(raw_text))
        if not text:
            continue

        image_id = str(row.get("image_id") or row.get("image") or "")
        image_name = str(row.get("image_name") or f"{image_id}.jpg")
        order = int(row.get("order") or row.get("index") or index)
        bbox_id = str(row.get("bbox_id") or f"img_{image_id}_bbox_{index:04d}")
        lines.append(
            OcrLine(
                bbox_id=bbox_id,
                image_id=image_id,
                image_name=image_name,
                order=order,
                det_score=det_score,
                text=text,
                text_raw=raw_text,
                bbox=parse_bbox(row),
            )
        )
    return sorted(lines, key=lambda line: (line.order, line.bbox_id))


def build_text_units(lines: list[OcrLine], tokenizer: Any, max_tokens: int) -> list[TextUnit]:
    if not lines:
        return []

    image_id = lines[0].image_id
    image_name = lines[0].image_name
    header = (
        f"Image ID: {image_id}\n"
        f"Image File: {image_name}\n"
        "OCR Transcript In Reading Order:\n"
    )

    units: list[TextUnit] = []
    current_lines: list[str] = []
    current_orders: list[int] = []

    def flush() -> None:
        if not current_lines:
            return
        unit_index = len(units) + 1
        body = "\n".join(current_lines)
        units.append(
            TextUnit(
                id=f"{image_id}_chunk_{unit_index:03d}",
                text=f"{header}{body}",
                line_count=len(current_lines),
                line_orders=current_orders.copy(),
            )
        )
        current_lines.clear()
        current_orders.clear()

    for line in lines:
        line_text = f"[{line.order}] {line.text}"
        candidate_lines = current_lines + [line_text]
        candidate_text = f"{header}{chr(10).join(candidate_lines)}"
        token_count = tokenizer.num_tokens(candidate_text)

        if current_lines and token_count > max_tokens:
            flush()

        current_lines.append(line_text)
        current_orders.append(line.order)

    flush()
    return units


def collect_image_ids(args: argparse.Namespace) -> list[str]:
    requested_ids = expand_image_ids(args.image_id)
    if requested_ids:
        image_ids = list(dict.fromkeys(requested_ids))
    elif args.all:
        image_ids = sorted(path.name for path in args.ocr_root.iterdir() if path.is_dir())
    else:
        raise SystemExit("Specify --image-id ... or use --all")

    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    return image_ids


def filter_orphan_relationships(relationships_df: pd.DataFrame, entities_df: pd.DataFrame) -> pd.DataFrame:
    if relationships_df.empty or entities_df.empty:
        return relationships_df.iloc[0:0].copy()
    entity_names = set(entities_df["title"].tolist())
    return relationships_df[
        relationships_df["source"].isin(entity_names)
        & relationships_df["target"].isin(entity_names)
    ].reset_index(drop=True)


def merge_entities(entity_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not entity_dfs:
        return pd.DataFrame(columns=["title", "type", "description", "text_unit_ids", "frequency"])
    all_entities = pd.concat(entity_dfs, ignore_index=True)
    return (
        all_entities.groupby(["title", "type"], sort=False)
        .agg(
            description=("description", list),
            text_unit_ids=("source_id", list),
            frequency=("source_id", "count"),
        )
        .reset_index()
    )


def merge_relationships(relationship_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not relationship_dfs:
        return pd.DataFrame(columns=["source", "target", "description", "text_unit_ids", "weight"])
    all_relationships = pd.concat(relationship_dfs, ignore_index=True)
    return (
        all_relationships.groupby(["source", "target"], sort=False)
        .agg(
            description=("description", list),
            text_unit_ids=("source_id", list),
            weight=("weight", "sum"),
        )
        .reset_index()
    )


async def summarize_entity_descriptions(
    entities_df: pd.DataFrame,
    summarizer: SummarizeExtractor | None,
) -> pd.DataFrame:
    if entities_df.empty:
        return entities_df

    records: list[dict[str, Any]] = []
    for row in entities_df.itertuples(index=False):
        descriptions = sorted(set(row.description))
        if summarizer is None or len(descriptions) <= 1:
            description = descriptions[0] if descriptions else ""
        else:
            result = await summarizer(id=str(row.title), descriptions=descriptions)
            description = result.description

        records.append(
            {
                "title": row.title,
                "type": row.type,
                "description": description,
                "descriptions_raw": descriptions,
                "text_unit_ids": list(dict.fromkeys(row.text_unit_ids)),
                "frequency": int(row.frequency),
            }
        )
    return pd.DataFrame(records)


async def summarize_relationship_descriptions(
    relationships_df: pd.DataFrame,
    summarizer: SummarizeExtractor | None,
) -> pd.DataFrame:
    if relationships_df.empty:
        return relationships_df

    records: list[dict[str, Any]] = []
    for row in relationships_df.itertuples(index=False):
        descriptions = sorted(set(row.description))
        if summarizer is None or len(descriptions) <= 1:
            description = descriptions[0] if descriptions else ""
        else:
            result = await summarizer(
                id=(str(row.source), str(row.target)),
                descriptions=descriptions,
            )
            description = result.description

        records.append(
            {
                "source": row.source,
                "target": row.target,
                "description": description,
                "descriptions_raw": descriptions,
                "text_unit_ids": list(dict.fromkeys(row.text_unit_ids)),
                "weight": float(row.weight),
            }
        )
    return pd.DataFrame(records)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_model(args: argparse.Namespace, api_key: str):
    model_config = ModelConfig(
        model_provider="gemini",
        model=args.model,
        api_key=api_key,
        auth_method=AuthMethod.ApiKey,
        retry={"type": "exponential_backoff", "max_retries": args.max_retries},
    )
    return create_completion(model_config)


async def extract_for_text_units(
    text_units: list[TextUnit],
    model: Any,
    entity_types: list[str],
    max_gleanings: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extractor = GraphExtractor(
        model=model,
        prompt=GRAPH_EXTRACTION_PROMPT_EN_VI_OUTPUT,
        max_gleanings=max_gleanings,
    )

    entity_dfs: list[pd.DataFrame] = []
    relationship_dfs: list[pd.DataFrame] = []
    for unit in text_units:
        entities_df, relationships_df = await extractor(
            unit.text,
            entity_types=entity_types,
            source_id=unit.id,
        )
        if not entities_df.empty:
            entity_dfs.append(entities_df)
        if not relationships_df.empty:
            relationship_dfs.append(relationships_df)

    entities_df = merge_entities(entity_dfs)
    relationships_df = merge_relationships(relationship_dfs)
    relationships_df = filter_orphan_relationships(relationships_df, entities_df)
    return entities_df, relationships_df


def build_graph_payload(
    image_id: str,
    image_name: str,
    ocr_dir: Path,
    lines: list[OcrLine],
    text_units: list[TextUnit],
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "graph_type": "graphrag_semantic_ocr_graph",
        "created_at": now_iso(),
        "image_id": image_id,
        "image_name": image_name,
        "model": {
            "provider": "gemini",
            "model": args.model,
            "max_retries": args.max_retries,
            "max_gleanings": args.max_gleanings,
            "entity_types": [item.strip() for item in args.entity_types.split(",") if item.strip()],
            "prompt_source": "graphrag.prompts.index.extract_graph.GRAPH_EXTRACTION_PROMPT + vi-output instruction",
            "summary_prompt_source": "graphrag.prompts.index.summarize_descriptions.SUMMARIZE_PROMPT + vi-output instruction",
        },
        "source_files": {
            "ocr_dir": str(ocr_dir),
            "ocr_file": str(find_ocr_file(ocr_dir) or ""),
        },
        "ocr_overview": {
            "line_count": len(lines),
            "non_empty_line_count": len(lines),
            "transcript_preview": [f"[{line.order}] {line.text}" for line in lines[:20]],
        },
        "text_units": [
            {
                "id": unit.id,
                "line_count": unit.line_count,
                "line_orders": unit.line_orders,
                "text": unit.text,
            }
            for unit in text_units
        ],
        "entity_count": int(len(entities_df)),
        "relationship_count": int(len(relationships_df)),
        "entities": entities_df.to_dict(orient="records"),
        "relationships": relationships_df.to_dict(orient="records"),
    }


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    entity_types = Counter(entity["type"] for entity in payload["entities"])
    relationship_weights = [float(item["weight"]) for item in payload["relationships"]] or [0.0]
    return {
        "created_at": payload["created_at"],
        "image_id": payload["image_id"],
        "image_name": payload["image_name"],
        "entity_count": payload["entity_count"],
        "relationship_count": payload["relationship_count"],
        "text_unit_count": len(payload["text_units"]),
        "entity_type_counts": dict(sorted(entity_types.items())),
        "max_relationship_weight": max(relationship_weights),
        "avg_relationship_weight": sum(relationship_weights) / len(relationship_weights),
    }


def try_load_resumed_summary(output_root: Path, image_id: str) -> dict[str, Any] | None:
    output_dir = output_root / image_id
    summary_path = output_dir / "summary.json"
    graph_path = output_dir / "graph.json"

    if summary_path.exists():
        payload = load_json(summary_path)
        if isinstance(payload, dict):
            return payload

    if graph_path.exists():
        payload = load_json(graph_path)
        if isinstance(payload, dict):
            summary = build_summary(payload)
            write_json(summary_path, summary)
            return summary

    return None


async def process_image(image_id: str, args: argparse.Namespace, model: Any) -> dict[str, Any] | None:
    ocr_dir = args.ocr_root / image_id
    rows = load_ocr_rows(ocr_dir)
    lines = rows_to_lines(rows, min_score=args.min_score)
    if not lines:
        return None

    text_units = build_text_units(
        lines=lines,
        tokenizer=model.tokenizer,
        max_tokens=args.max_text_unit_tokens,
    )
    if not text_units:
        return None

    entity_types = [item.strip() for item in args.entity_types.split(",") if item.strip()]
    entities_df, relationships_df = await extract_for_text_units(
        text_units=text_units,
        model=model,
        entity_types=entity_types,
        max_gleanings=args.max_gleanings,
    )
    if entities_df.empty:
        raise ValueError(
            f"Graph extraction returned no entities for image {image_id}. "
            "This usually means the model call failed or the OCR text had no extractable entities."
        )
    if relationships_df.empty:
        raise ValueError(
            f"Graph extraction returned no relationships for image {image_id}. "
            "This usually means the model call failed or the OCR text lacked clear relations."
        )

    summarizer = None
    if not args.no_summary:
        summarizer = SummarizeExtractor(
            model=model,
            max_summary_length=args.summary_max_length,
            max_input_tokens=args.summary_max_input_tokens,
            summarization_prompt=SUMMARIZE_PROMPT_EN_VI_OUTPUT,
        )

    entities_df = await summarize_entity_descriptions(entities_df, summarizer)
    relationships_df = await summarize_relationship_descriptions(relationships_df, summarizer)

    payload = build_graph_payload(
        image_id=image_id,
        image_name=lines[0].image_name,
        ocr_dir=ocr_dir,
        lines=lines,
        text_units=text_units,
        entities_df=entities_df,
        relationships_df=relationships_df,
        args=args,
    )
    summary = build_summary(payload)

    output_dir = args.output_root / image_id
    write_json(output_dir / "graph.json", payload)
    write_json(output_dir / "summary.json", summary)
    return summary


async def main_async() -> None:
    args = parse_args()
    image_ids = collect_image_ids(args)
    model = None

    manifest = {
        "created_at": now_iso(),
        "ocr_root": str(args.ocr_root),
        "output_root": str(args.output_root),
        "model": args.model,
        "max_retries": args.max_retries,
        "entity_types": [item.strip() for item in args.entity_types.split(",") if item.strip()],
        "max_gleanings": args.max_gleanings,
        "max_text_unit_tokens": args.max_text_unit_tokens,
        "resume": args.resume,
        "sleep_seconds": args.sleep_seconds,
        "dry_run": args.dry_run,
        "requested_image_ids": image_ids,
    }

    if args.dry_run:
        write_json(args.output_root / "dry_run.json", manifest)
        print(
            f"Dry run complete for {len(image_ids)} image(s). "
            f"Manifest written to {args.output_root.resolve()}"
        )
        return

    summaries: list[dict[str, Any]] = []
    processed_summaries: list[dict[str, Any]] = []
    resumed_image_ids: list[str] = []
    skipped: list[str] = []
    failures: dict[str, str] = {}

    progress = tqdm(
        image_ids,
        desc="Building semantic graphs",
        unit="image",
        dynamic_ncols=True,
    )
    for image_id in progress:
        if args.resume:
            resumed_summary = try_load_resumed_summary(args.output_root, image_id)
            if resumed_summary is not None:
                summaries.append(resumed_summary)
                resumed_image_ids.append(image_id)
                progress.set_postfix(
                    processed=len(processed_summaries),
                    resumed=len(resumed_image_ids),
                    skipped=len(skipped),
                    failed=len(failures),
                )
                continue

        if model is None:
            api_key = resolve_api_key(args)
            model = build_model(args, api_key)

        try:
            summary = await process_image(image_id, args, model)
        except Exception as exc:  # pragma: no cover - defensive logging
            failures[image_id] = f"{type(exc).__name__}: {exc}"
            progress.set_postfix(
                processed=len(processed_summaries),
                resumed=len(resumed_image_ids),
                skipped=len(skipped),
                failed=len(failures),
            )
            if args.sleep_seconds > 0:
                await asyncio.sleep(args.sleep_seconds)
            continue

        if summary is None:
            skipped.append(image_id)
            progress.set_postfix(
                processed=len(processed_summaries),
                resumed=len(resumed_image_ids),
                skipped=len(skipped),
                failed=len(failures),
            )
            if args.sleep_seconds > 0:
                await asyncio.sleep(args.sleep_seconds)
            continue
        summaries.append(summary)
        processed_summaries.append(summary)
        progress.set_postfix(
            processed=len(processed_summaries),
            resumed=len(resumed_image_ids),
            skipped=len(skipped),
            failed=len(failures),
        )

        if args.sleep_seconds > 0:
            await asyncio.sleep(args.sleep_seconds)

    manifest.update(
        {
            "processed_count": len(processed_summaries),
            "resumed_count": len(resumed_image_ids),
            "resumed_image_ids": resumed_image_ids,
            "completed_count": len(summaries),
            "skipped_count": len(skipped),
            "skipped_image_ids": skipped,
            "failed_count": len(failures),
            "failures": failures,
            "summaries": summaries,
        }
    )
    write_json(args.output_root / "build_summary.json", manifest)

    print(
        f"Processed {len(processed_summaries)} image(s), resumed {len(resumed_image_ids)}, "
        f"skipped {len(skipped)}, failed {len(failures)}. "
        f"Graphs written to {args.output_root.resolve()}"
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
