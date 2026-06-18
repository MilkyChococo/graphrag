from __future__ import annotations

import argparse
import asyncio
import json
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
    resolve_api_key,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_FILE = REPO_ROOT / "test_v1.0.json"
DEFAULT_IMAGES_ROOT = REPO_ROOT
DEFAULT_SEMANTIC_ROOT = SCRIPT_DIR / "semantic_graphs"
DEFAULT_BYOG_ROOT = SCRIPT_DIR / "byog_workspaces"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "qwen_predictions"

SYSTEM_PROMPT = """You answer SP-DocVQA questions using OCR graph context.
Return only the final short answer. Do not explain.
Use only the provided context. Preserve exact OCR strings when possible.
If the context is insufficient, return: Not enough information"""

USER_PROMPT_TEMPLATE = """Question: {question}

OCR and graph context:
{context}

Short answer:"""

IMAGE_ONLY_PROMPT_TEMPLATE = """Question: {question}

Answer using the visible information in the image.
Return only the final short answer."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SP-DocVQA baseline inference with local Transformers Qwen2.5-VL-7B-Instruct over semantic/BYOG context."
    )
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--byog-root", type=Path, default=DEFAULT_BYOG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--image-detail", choices=["auto", "low", "high"], default="high")
    parser.add_argument("--max-context-chars", type=int, default=12000)
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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def image_stem(image_value: str) -> str:
    return Path(str(image_value)).stem


def resolve_image_path(images_root: Path, image_value: str) -> Path:
    return (images_root / str(image_value)).resolve()


def load_completed(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["questionId"])] = record
    return records


def build_context(graph: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    preview = graph.get("ocr_overview", {}).get("transcript_preview", [])
    if preview:
        parts.append("OCR preview:\n" + "\n".join(str(line) for line in preview))

    text_units = graph.get("text_units", [])
    if text_units:
        parts.append(
            "Text units:\n"
            + "\n\n".join(str(unit.get("text", "")) for unit in text_units[:4] if unit.get("text"))
        )

    entities = graph.get("entities", [])
    if entities:
        parts.append(
            "Entities:\n"
            + "\n".join(
                f"- {item.get('title')} ({item.get('type')}): {item.get('description')}"
                for item in entities[:80]
            )
        )

    relationships = graph.get("relationships", [])
    if relationships:
        parts.append(
            "Relationships:\n"
            + "\n".join(
                f"- {item.get('source')} -> {item.get('target')}: {item.get('description')}"
                for item in relationships[:80]
            )
        )

    context = "\n\n".join(part for part in parts if part).strip()
    if len(context) > max_chars:
        return context[: max_chars - 3].rstrip() + "..."
    return context


def build_submission(source_payload: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_name": source_payload.get("dataset_name"),
        "dataset_version": source_payload.get("dataset_version"),
        "dataset_split": source_payload.get("dataset_split"),
        "data": [
            {"questionId": int(record["questionId"]), "answer": record["answer"]}
            for record in sorted(records, key=lambda item: int(item["questionId"]))
        ],
    }


async def main_async() -> None:
    args = parse_args()
    source_payload = load_json(args.test_file.resolve())
    questions = list(source_payload.get("data", []))
    if args.limit is not None:
        questions = questions[: args.limit]

    output_jsonl = args.output_root / "docvqa_qwen_predictions.jsonl"
    output_submission = args.output_root / "docvqa_qwen_submission.json"
    output_summary = args.output_root / "docvqa_qwen_summary.json"
    if not args.resume and output_jsonl.exists():
        output_jsonl.write_text("", encoding="utf-8")

    completed = load_completed(output_jsonl) if args.resume else {}
    records = list(completed.values())
    api_key = resolve_api_key(args.env_file, args.api_key_env)
    model = build_completion_model(
        provider=args.provider,
        model_name=args.model,
        api_key=api_key,
        api_base=args.api_base,
        max_retries=args.max_retries,
    )

    missing_semantic = 0
    missing_image = 0
    processed = 0
    for row in tqdm(questions, desc="Qwen DocVQA inference", unit="question"):
        question_id = int(row["questionId"])
        if question_id in completed:
            continue
        image_id = image_stem(row["image"])
        image_path = resolve_image_path(args.images_root, str(row["image"]))
        graph_path = args.semantic_root / image_id / "graph.json"
        workspace_root = args.byog_root / image_id
        if not graph_path.exists():
            missing_semantic += 1
            prompt = IMAGE_ONLY_PROMPT_TEMPLATE.format(question=row["question"])
            image_used = image_path.exists()
            if image_used:
                answer = await complete_with_image(
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    image_path=image_path,
                    image_detail=args.image_detail,
                    temperature=args.temperature,
                )
            else:
                missing_image += 1
                answer = await complete_text(
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=args.temperature,
                )
            detail = {
                "reason": "missing_semantic_graph_image_only" if image_used else "missing_semantic_graph_text_only",
                "graph_path": str(graph_path),
                "image_path": str(image_path),
                "image_used": image_used,
                "workspace_root": str(workspace_root),
                "byog_exists": workspace_root.exists(),
            }
        else:
            graph = load_json(graph_path)
            context = build_context(graph, args.max_context_chars)
            prompt = USER_PROMPT_TEMPLATE.format(question=row["question"], context=context)
            image_used = image_path.exists()
            if image_used:
                answer = await complete_with_image(
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    image_path=image_path,
                    image_detail=args.image_detail,
                    temperature=args.temperature,
                )
            else:
                answer = await complete_text(
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=args.temperature,
                )
            detail = {
                "reason": "qwen_vl_image_semantic_byog_context" if image_used else "qwen_text_semantic_byog_context",
                "graph_path": str(graph_path),
                "image_path": str(image_path),
                "image_used": image_used,
                "workspace_root": str(workspace_root),
                "byog_exists": workspace_root.exists(),
            }

        record = {
            "questionId": question_id,
            "image_id": image_id,
            "image": row.get("image"),
            "question": row.get("question"),
            "answer": str(answer).strip().splitlines()[0] if str(answer).strip() else "",
            "method": "qwen2_5_vl_7b_hf_semantic_byog_baseline",
            "model": args.model,
            "provider": args.provider,
            "detail": detail,
            "created_at": now_iso(),
        }
        append_jsonl(output_jsonl, record)
        records.append(record)
        processed += 1
        if args.sleep_seconds > 0:
            await asyncio.sleep(args.sleep_seconds)

    write_json(output_submission, build_submission(source_payload, records))
    write_json(
        output_summary,
        {
            "created_at": now_iso(),
            "test_file": str(args.test_file.resolve()),
            "semantic_root": str(args.semantic_root.resolve()),
            "images_root": str(args.images_root.resolve()),
            "byog_root": str(args.byog_root.resolve()),
            "output_jsonl": str(output_jsonl.resolve()),
            "output_submission": str(output_submission.resolve()),
            "requested_question_count": len(questions),
            "processed_count": processed,
            "resumed_count": len(completed),
            "record_count": len(records),
            "missing_semantic_count": missing_semantic,
            "missing_image_count": missing_image,
            "method": "qwen2_5_vl_7b_hf_semantic_byog_baseline",
            "model": args.model,
            "provider": args.provider,
        },
    )
    print(
        f"Prepared Qwen inference outputs at {args.output_root.resolve()} "
        f"(processed={processed}, resumed={len(completed)}, missing_semantic={missing_semantic})"
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
