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

import answer_with_graph_or_vlm as hybrid


DEFAULT_ANNOTATIONS_FILE = SCRIPT_DIR.parent / "ViTextVQA_images" / "ViTextVQA_test.json"
DEFAULT_OUTPUT_JSONL = SCRIPT_DIR / "hybrid_test_predictions.jsonl"
DEFAULT_OUTPUT_SUBMISSION = SCRIPT_DIR / "hybrid_test_submission.json"
DEFAULT_OUTPUT_SUMMARY = SCRIPT_DIR / "hybrid_test_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full ViTextVQA test inference with GraphRAG when available and VLM fallback otherwise."
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
        default=hybrid.DEFAULT_BYOG_ROOT,
        help=f"Root containing per-image BYOG workspaces. Default: {hybrid.DEFAULT_BYOG_ROOT}",
    )
    parser.add_argument(
        "--semantic-root",
        type=Path,
        default=hybrid.DEFAULT_SEMANTIC_ROOT,
        help=f"Root containing per-image semantic graph JSON files. Default: {hybrid.DEFAULT_SEMANTIC_ROOT}",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=hybrid.DEFAULT_IMAGES_ROOT,
        help=f"Root containing image files. Default: {hybrid.DEFAULT_IMAGES_ROOT}",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=hybrid.DEFAULT_ENV_FILE,
        help=f".env file used to resolve API keys for litellm_api fallback. Default: {hybrid.DEFAULT_ENV_FILE}",
    )
    parser.add_argument(
        "--vlm-backend",
        choices=["local_qwen", "litellm_api"],
        default=hybrid.DEFAULT_VLM_BACKEND,
        help=f"VLM backend used for fallback. Default: {hybrid.DEFAULT_VLM_BACKEND}",
    )
    parser.add_argument(
        "--vlm-provider",
        default=hybrid.DEFAULT_VLM_PROVIDER,
        help=f"LiteLLM provider used when --vlm-backend=litellm_api. Default: {hybrid.DEFAULT_VLM_PROVIDER}",
    )
    parser.add_argument(
        "--vlm-api-base",
        default=None,
        help="Optional LiteLLM api_base for OpenAI-compatible / proxy endpoints.",
    )
    parser.add_argument(
        "--api-key-env",
        default="LITELLM_API_KEY",
        help="Environment variable holding the API key when --vlm-backend=litellm_api.",
    )
    parser.add_argument(
        "--vlm-model",
        default=hybrid.DEFAULT_VLM_MODEL,
        help=f"VLM model used for fallback. Default: {hybrid.DEFAULT_VLM_MODEL}",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for model/API calls. Default: 3",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=hybrid.DEFAULT_VLM_TEMPERATURE,
        help=f"Sampling temperature for VLM fallback. Default: {hybrid.DEFAULT_VLM_TEMPERATURE}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=hybrid.DEFAULT_VLM_MAX_NEW_TOKENS,
        help=f"Maximum new tokens for local_qwen fallback. Default: {hybrid.DEFAULT_VLM_MAX_NEW_TOKENS}",
    )
    parser.add_argument(
        "--community-level",
        type=int,
        default=2,
        help="GraphRAG community level. Default: 2",
    )
    parser.add_argument(
        "--query-method",
        choices=["basic", "local", "global", "drift"],
        default=hybrid.DEFAULT_QUERY_METHOD,
        help=f"GraphRAG query method. Default: {hybrid.DEFAULT_QUERY_METHOD}",
    )
    parser.add_argument(
        "--fallback-query-method",
        choices=["none", "basic", "local", "global", "drift"],
        default=hybrid.DEFAULT_FALLBACK_QUERY_METHOD,
        help=f"Fallback GraphRAG query method. Default: {hybrid.DEFAULT_FALLBACK_QUERY_METHOD}",
    )
    parser.add_argument(
        "--response-type",
        default=hybrid.DEFAULT_RESPONSE_TYPE,
        help=f"GraphRAG response type. Default: {hybrid.DEFAULT_RESPONSE_TYPE}",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Do not auto-build missing GraphRAG query artifacts before querying.",
    )
    parser.add_argument(
        "--graphrag-backend",
        choices=["local_hf", "graphrag_api"],
        default=hybrid.DEFAULT_GRAPHRAG_BACKEND,
        help=f"GraphRAG inference backend. Default: {hybrid.DEFAULT_GRAPHRAG_BACKEND}",
    )
    parser.add_argument(
        "--graphrag-completion-model",
        default=hybrid.DEFAULT_GRAPHRAG_COMPLETION_MODEL,
        help=f"Local Hugging Face generation model used when --graphrag-backend=local_hf. Default: {hybrid.DEFAULT_GRAPHRAG_COMPLETION_MODEL}",
    )
    parser.add_argument(
        "--graphrag-embedding-model",
        default=hybrid.DEFAULT_GRAPHRAG_EMBEDDING_MODEL,
        help=f"Local Hugging Face embedding model used when --graphrag-backend=local_hf. Default: {hybrid.DEFAULT_GRAPHRAG_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=hybrid.DEFAULT_RETRIEVAL_TOP_K,
        help=f"Number of retrieved graph/OCR context items to feed into local_hf GraphRAG. Default: {hybrid.DEFAULT_RETRIEVAL_TOP_K}",
    )
    parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default=hybrid.DEFAULT_EMBEDDING_DEVICE,
        help=f"Device for local embedding model. Default: {hybrid.DEFAULT_EMBEDDING_DEVICE}",
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
            annotation_id = int(record["annotation_id"])
            completed_ids.add(annotation_id)
    return completed_ids, records


def is_success_record(record: dict[str, Any]) -> bool:
    if "success" in record:
        return bool(record["success"])
    return bool(str(record.get("answer", "")).strip())


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
    byog_root = args.byog_root.resolve()
    semantic_root = args.semantic_root.resolve()
    images_root = args.images_root.resolve()
    fallback_query_method = (
        None if args.fallback_query_method == "none" else args.fallback_query_method
    )
    if args.limit is not None:
        annotations = annotations[: args.limit]

    completed_ids: set[int] = set()
    existing_records: list[dict[str, Any]] = []
    if args.resume:
        completed_ids, existing_records = load_completed_records(args.output_jsonl.resolve())

    annotation_records = [ann for ann in annotations if int(ann["id"]) not in completed_ids]

    if existing_records:
        write_submission_checkpoint(
            source_payload=source_payload,
            records=existing_records,
            output_path=args.output_submission.resolve(),
        )

    workspace_status_cache: dict[str, Path] = {}
    qa_cache: dict[tuple[str, str], dict[str, Any]] = {}

    method_counts: Counter[str] = Counter(record.get("used_method", "unknown") for record in existing_records)
    query_method_counts: Counter[str] = Counter(
        record.get("query_method_used", "unknown")
        for record in existing_records
        if record.get("query_method_used")
    )
    fallback_counts: Counter[str] = Counter()
    graph_workspace_hits = 0
    graph_workspace_misses = 0
    index_builds = 0
    cache_hits = 0
    success_count = sum(1 for record in existing_records if is_success_record(record))
    fail_count = len(existing_records) - success_count

    progress = tqdm(
        annotation_records,
        desc="Hybrid test inference",
        unit="question",
        dynamic_ncols=True,
    )

    for ann in progress:
        annotation_id = int(ann["id"])
        image_id = str(ann["image_id"])
        question = str(ann["question"])
        image_path = hybrid.resolve_image_path(images_root, image_id)
        cache_key = (image_id, question)

        if cache_key in qa_cache:
            cached = qa_cache[cache_key]
            payload = {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "question": question,
                "answer": cached["answer"],
                "used_method": cached["used_method"],
                "success": cached.get("success", bool(str(cached["answer"]).strip())),
                "error": cached.get("error"),
                "cache_hit": True,
                "fallback_reason": cached.get("fallback_reason"),
                "workspace_root": cached.get("workspace_root"),
                "image_path": str(image_path),
                "query_method_requested": cached.get("query_method_requested", args.query_method),
                "query_method_used": cached.get("query_method_used"),
                "index_built": cached.get("index_built", False),
                "graphrag_backend": cached.get("graphrag_backend", args.graphrag_backend),
                "vlm_backend": cached.get("vlm_backend", args.vlm_backend),
                "vlm_model": cached.get("vlm_model", args.vlm_model),
                "created_at": now_iso(),
            }
            append_jsonl(args.output_jsonl.resolve(), payload)
            existing_records.append(payload)
            write_submission_checkpoint(
                source_payload=source_payload,
                records=existing_records,
                output_path=args.output_submission.resolve(),
            )
            method_counts[payload["used_method"]] += 1
            cache_hits += 1
            if payload["success"]:
                success_count += 1
            else:
                fail_count += 1
            progress.set_postfix(
                graphrag=method_counts["graphrag"],
                vlm=method_counts["vlm"],
                success=success_count,
                fail=fail_count,
                cache=cache_hits,
            )
            continue

        workspace_root = workspace_status_cache.get(image_id, Path())
        if not workspace_root:
            workspace_root = hybrid.resolve_workspace(byog_root, image_id)
            workspace_status_cache[image_id] = workspace_root

        answer = ""
        used_method = ""
        fallback_reason: str | None = None
        query_method_used: str | None = None
        index_built = False
        error_message: str | None = None
        success = False

        try:
            if workspace_root.exists():
                graph_workspace_hits += 1
                try:
                    graphrag_payload = await hybrid.answer_with_graphrag(
                        image_id=image_id,
                        question=question,
                        byog_root=byog_root,
                        semantic_root=semantic_root,
                        images_root=images_root,
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
                    answer = graphrag_payload["answer"]
                    used_method = graphrag_payload["used_method"]
                    fallback_reason = graphrag_payload.get("fallback_reason")
                    query_method_used = graphrag_payload.get("query_method_used")
                    index_built = bool(graphrag_payload.get("index_built"))
                    if index_built:
                        index_builds += 1
                except Exception as exc:  # pragma: no cover - runtime fallback
                    fallback_reason = f"GraphRAG query failed: {type(exc).__name__}: {exc}"
            else:
                graph_workspace_misses += 1
                fallback_reason = "Per-image BYOG workspace not found"

            if not answer:
                answer = await hybrid.run_vlm_query(
                    image_path=image_path,
                    question=question,
                    backend=args.vlm_backend,
                    model_name=args.vlm_model,
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    provider=args.vlm_provider,
                    api_base=args.vlm_api_base,
                    env_file=args.env_file,
                    api_key_env=args.api_key_env,
                )
                used_method = "vlm"
                if fallback_reason:
                    fallback_counts[fallback_reason] += 1
            success = bool(str(answer).strip())
        except Exception as exc:  # pragma: no cover - keep full-run resilient
            used_method = used_method or "failed"
            answer = ""
            success = False
            error_message = f"{type(exc).__name__}: {exc}"

        payload = {
            "annotation_id": annotation_id,
            "image_id": image_id,
            "question": question,
            "answer": answer,
            "used_method": used_method,
            "success": success,
            "error": error_message,
            "cache_hit": False,
            "fallback_reason": fallback_reason,
            "workspace_root": str(workspace_root),
            "image_path": str(image_path),
            "query_method_requested": args.query_method,
            "query_method_used": query_method_used,
            "index_built": index_built,
            "graphrag_backend": args.graphrag_backend,
            "vlm_backend": args.vlm_backend,
            "vlm_model": args.vlm_model,
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
            "answer": answer,
            "used_method": used_method,
            "success": success,
            "error": error_message,
            "fallback_reason": fallback_reason,
            "workspace_root": str(workspace_root),
            "query_method_requested": args.query_method,
            "query_method_used": query_method_used,
            "index_built": index_built,
            "graphrag_backend": args.graphrag_backend,
            "vlm_backend": args.vlm_backend,
            "vlm_model": args.vlm_model,
        }
        method_counts[used_method] += 1
        if query_method_used:
            query_method_counts[query_method_used] += 1
        if success:
            success_count += 1
        else:
            fail_count += 1
        progress.set_postfix(
            graphrag=method_counts["graphrag"],
            vlm=method_counts["vlm"],
            success=success_count,
            fail=fail_count,
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
        "vlm_backend": args.vlm_backend,
        "vlm_model": args.vlm_model,
        "requested_count": len(annotations),
        "completed_count": len(existing_records),
        "newly_processed_count": len(annotation_records),
        "resumed_count": len(completed_ids),
        "success_count": success_count,
        "fail_count": fail_count,
        "method_counts": dict(method_counts),
        "query_method_counts": dict(query_method_counts),
        "graph_workspace_hits": graph_workspace_hits,
        "graph_workspace_misses": graph_workspace_misses,
        "index_builds": index_builds,
        "cache_hits": cache_hits,
        "fallback_counts": dict(fallback_counts),
    }
    write_json(args.output_summary.resolve(), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
