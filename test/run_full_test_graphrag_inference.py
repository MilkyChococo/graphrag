from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import answer_with_graph_or_vlm as graphrag_answer


DEFAULT_ANNOTATIONS_FILE = SCRIPT_DIR.parent / "ViTextVQA_images" / "ViTextVQA_test.json"
DEFAULT_OUTPUT_JSONL = SCRIPT_DIR / "graphrag_test_predictions.jsonl"
DEFAULT_OUTPUT_SUBMISSION = SCRIPT_DIR / "graphrag_test_submission.json"
DEFAULT_OUTPUT_SUMMARY = SCRIPT_DIR / "graphrag_test_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full ViTextVQA test inference with GraphRAG only."
    )
    parser.add_argument(
        "--annotations-file",
        type=Path,
        default=DEFAULT_ANNOTATIONS_FILE,
        help=f"Question file in ViTextVQA format. Default: {DEFAULT_ANNOTATIONS_FILE}",
    )
    parser.add_argument(
        "--byog-root",
        type=Path,
        default=graphrag_answer.DEFAULT_BYOG_ROOT,
        help=f"Root containing per-image BYOG workspaces. Default: {graphrag_answer.DEFAULT_BYOG_ROOT}",
    )
    parser.add_argument(
        "--semantic-root",
        type=Path,
        default=graphrag_answer.DEFAULT_SEMANTIC_ROOT,
        help=f"Root containing per-image semantic graph JSON files. Default: {graphrag_answer.DEFAULT_SEMANTIC_ROOT}",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=graphrag_answer.DEFAULT_IMAGES_ROOT,
        help=f"Root containing image files. Default: {graphrag_answer.DEFAULT_IMAGES_ROOT}",
    )
    parser.add_argument(
        "--query-method",
        choices=["basic", "local", "global", "drift"],
        default=graphrag_answer.DEFAULT_QUERY_METHOD,
        help=f"Primary GraphRAG query method. Default: {graphrag_answer.DEFAULT_QUERY_METHOD}",
    )
    parser.add_argument(
        "--fallback-query-method",
        choices=["none", "basic", "local", "global", "drift"],
        default=graphrag_answer.DEFAULT_FALLBACK_QUERY_METHOD,
        help=f"Fallback GraphRAG query method. Default: {graphrag_answer.DEFAULT_FALLBACK_QUERY_METHOD}",
    )
    parser.add_argument(
        "--community-level",
        type=int,
        default=2,
        help="GraphRAG community level. Default: 2",
    )
    parser.add_argument(
        "--response-type",
        default=graphrag_answer.DEFAULT_RESPONSE_TYPE,
        help=f"GraphRAG response type. Default: {graphrag_answer.DEFAULT_RESPONSE_TYPE}",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help=f"Detailed prediction log. Default: {DEFAULT_OUTPUT_JSONL}",
    )
    parser.add_argument(
        "--output-submission",
        type=Path,
        default=DEFAULT_OUTPUT_SUBMISSION,
        help=f"Submission-style JSON output. Default: {DEFAULT_OUTPUT_SUBMISSION}",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
        help=f"Run summary JSON output. Default: {DEFAULT_OUTPUT_SUMMARY}",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --output-jsonl by skipping completed annotation ids.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of annotations to process.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between uncached model-backed inferences. Default: 0.0",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Do not auto-build missing GraphRAG query artifacts before querying.",
    )
    parser.add_argument(
        "--graphrag-backend",
        choices=["local_hf", "graphrag_api"],
        default=graphrag_answer.DEFAULT_GRAPHRAG_BACKEND,
        help=f"GraphRAG inference backend. Default: {graphrag_answer.DEFAULT_GRAPHRAG_BACKEND}",
    )
    parser.add_argument(
        "--graphrag-completion-model",
        default=graphrag_answer.DEFAULT_GRAPHRAG_COMPLETION_MODEL,
        help=f"Local Hugging Face generation model used when --graphrag-backend=local_hf. Default: {graphrag_answer.DEFAULT_GRAPHRAG_COMPLETION_MODEL}",
    )
    parser.add_argument(
        "--graphrag-embedding-model",
        default=graphrag_answer.DEFAULT_GRAPHRAG_EMBEDDING_MODEL,
        help=f"Local Hugging Face embedding model used when --graphrag-backend=local_hf. Default: {graphrag_answer.DEFAULT_GRAPHRAG_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=graphrag_answer.DEFAULT_RETRIEVAL_TOP_K,
        help=f"Number of retrieved graph/OCR context items to feed into local_hf GraphRAG. Default: {graphrag_answer.DEFAULT_RETRIEVAL_TOP_K}",
    )
    parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default=graphrag_answer.DEFAULT_EMBEDDING_DEVICE,
        help=f"Device for local embedding model. Default: {graphrag_answer.DEFAULT_EMBEDDING_DEVICE}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=graphrag_answer.DEFAULT_VLM_TEMPERATURE,
        help=f"Sampling temperature for local_hf GraphRAG answer generation. Default: {graphrag_answer.DEFAULT_VLM_TEMPERATURE}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=graphrag_answer.DEFAULT_VLM_MAX_NEW_TOKENS,
        help=f"Maximum new tokens for local_hf GraphRAG answer generation. Default: {graphrag_answer.DEFAULT_VLM_MAX_NEW_TOKENS}",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_annotations(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    if not isinstance(payload, dict) or not isinstance(annotations, list):
        raise SystemExit(f"Invalid ViTextVQA annotation file: {path}")
    return payload, annotations


def load_completed_records(path: Path) -> tuple[set[int], list[dict[str, Any]]]:
    completed_ids: set[int] = set()
    records: list[dict[str, Any]] = []
    if not path.exists():
        return completed_ids, records

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            completed_ids.add(int(record["annotation_id"]))
    return completed_ids, records


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def build_submission(
    source_payload: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    records_by_id = {int(record["annotation_id"]): record for record in records}
    submission_annotations: list[dict[str, Any]] = []
    for ann in source_payload["annotations"]:
        ann_id = int(ann["id"])
        record = records_by_id.get(ann_id)
        if record is None:
            continue
        submission_annotations.append(
            {
                "id": ann_id,
                "image_id": int(ann["image_id"]),
                "question": ann["question"],
                "answers": [record["answer"]],
            }
        )
    return {
        "images": source_payload["images"],
        "annotations": submission_annotations,
    }


def write_submission_checkpoint(
    source_payload: dict[str, Any],
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    submission = build_submission(source_payload, records)
    write_json(output_path, submission)


async def main_async() -> None:
    args = parse_args()
    source_payload, annotations = load_annotations(args.annotations_file.resolve())
    if args.limit is not None:
        annotations = annotations[: args.limit]

    completed_ids: set[int] = set()
    existing_records: list[dict[str, Any]] = []
    if args.resume:
        completed_ids, existing_records = load_completed_records(args.output_jsonl.resolve())

    annotation_records = [ann for ann in annotations if int(ann["id"]) not in completed_ids]
    fallback_query_method = (
        None if args.fallback_query_method == "none" else args.fallback_query_method
    )

    if existing_records:
        write_submission_checkpoint(
            source_payload=source_payload,
            records=existing_records,
            output_path=args.output_submission.resolve(),
        )

    qa_cache: dict[tuple[str, str], dict[str, Any]] = {}
    used_method_counts: Counter[str] = Counter(
        record.get("used_method", "unknown") for record in existing_records
    )
    query_method_counts: Counter[str] = Counter(
        record.get("query_method_used", "unknown") for record in existing_records
    )
    cache_hits = 0
    index_builds = sum(1 for record in existing_records if record.get("index_built"))

    progress = tqdm(
        annotation_records,
        desc="GraphRAG test inference",
        unit="question",
        dynamic_ncols=True,
    )

    for ann in progress:
        annotation_id = int(ann["id"])
        image_id = str(ann["image_id"])
        question = str(ann["question"])
        cache_key = (image_id, question)

        if cache_key in qa_cache:
            cached = qa_cache[cache_key]
            payload = {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "question": question,
                "answer": cached["answer"],
                "used_method": cached["used_method"],
                "query_method_requested": args.query_method,
                "query_method_used": cached["query_method_used"],
                "cache_hit": True,
                "fallback_reason": cached.get("fallback_reason"),
                "workspace_root": cached.get("workspace_root"),
                "index_built": False,
                "created_at": now_iso(),
            }
            append_jsonl(args.output_jsonl.resolve(), payload)
            existing_records.append(payload)
            write_submission_checkpoint(
                source_payload=source_payload,
                records=existing_records,
                output_path=args.output_submission.resolve(),
            )
            used_method_counts[payload["used_method"]] += 1
            query_method_counts[payload["query_method_used"]] += 1
            cache_hits += 1
            progress.set_postfix(
                graphrag=used_method_counts["graphrag"],
                local=query_method_counts["local"],
                basic=query_method_counts["basic"],
                cache=cache_hits,
            )
            continue

        result = await graphrag_answer.answer_with_graphrag(
            image_id=image_id,
            question=question,
            byog_root=args.byog_root.resolve(),
            semantic_root=args.semantic_root.resolve(),
            images_root=args.images_root.resolve(),
            query_method=args.query_method,
            fallback_query_method=fallback_query_method,
            community_level=args.community_level,
            response_type=args.response_type,
            auto_index=not args.skip_index,
            backend=args.graphrag_backend,
            completion_model_name=args.graphrag_completion_model,
            embedding_model_name=args.graphrag_embedding_model,
            retrieval_top_k=args.retrieval_top_k,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            embedding_device=args.embedding_device,
        )

        payload = {
            "annotation_id": annotation_id,
            "image_id": image_id,
            "question": question,
            "answer": result["answer"],
            "used_method": result["used_method"],
            "query_method_requested": args.query_method,
            "query_method_used": result["query_method_used"],
            "cache_hit": False,
            "fallback_reason": result.get("fallback_reason"),
            "workspace_root": result["workspace_root"],
            "index_built": bool(result.get("index_built")),
            "graphrag_backend": result.get("graphrag_backend", args.graphrag_backend),
            "created_at": now_iso(),
        }
        append_jsonl(args.output_jsonl.resolve(), payload)
        existing_records.append(payload)
        write_submission_checkpoint(
            source_payload=source_payload,
            records=existing_records,
            output_path=args.output_submission.resolve(),
        )
        qa_cache[cache_key] = {
            "answer": payload["answer"],
            "used_method": payload["used_method"],
            "query_method_used": payload["query_method_used"],
            "fallback_reason": payload["fallback_reason"],
            "workspace_root": payload["workspace_root"],
        }
        used_method_counts[payload["used_method"]] += 1
        query_method_counts[payload["query_method_used"]] += 1
        if payload["index_built"]:
            index_builds += 1
        progress.set_postfix(
            graphrag=used_method_counts["graphrag"],
            local=query_method_counts["local"],
            basic=query_method_counts["basic"],
            cache=cache_hits,
        )

        if args.sleep_seconds > 0:
            await asyncio.sleep(args.sleep_seconds)

    write_submission_checkpoint(
        source_payload=source_payload,
        records=existing_records,
        output_path=args.output_submission.resolve(),
    )

    summary = {
        "created_at": now_iso(),
        "annotations_file": str(args.annotations_file.resolve()),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "output_submission": str(args.output_submission.resolve()),
        "query_method_requested": args.query_method,
        "fallback_query_method": fallback_query_method,
        "graphrag_backend": args.graphrag_backend,
        "graphrag_completion_model": args.graphrag_completion_model,
        "graphrag_embedding_model": args.graphrag_embedding_model,
        "requested_count": len(annotations),
        "completed_count": len(existing_records),
        "newly_processed_count": len(annotation_records),
        "resumed_count": len(completed_ids),
        "used_method_counts": dict(used_method_counts),
        "query_method_counts": dict(query_method_counts),
        "cache_hits": cache_hits,
        "index_builds": index_builds,
    }
    write_json(args.output_summary.resolve(), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
