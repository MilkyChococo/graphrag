from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import mimetypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGES_ROOT = REPO_ROOT / "packages"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if PACKAGES_ROOT.exists():
    for package_dir in sorted(PACKAGES_ROOT.iterdir()):
        if package_dir.is_dir() and str(package_dir) not in sys.path:
            sys.path.insert(0, str(package_dir))

DEFAULT_ENV_FILE = SCRIPT_DIR.parent / ".env"
DEFAULT_IMAGES_ROOT = SCRIPT_DIR.parent / "ViTextVQA_images" / "st_images"
DEFAULT_BYOG_ROOT = SCRIPT_DIR / "graphrag_baseline_per_image"
DEFAULT_VLM_BACKEND = "local_qwen"
DEFAULT_VLM_PROVIDER = "gemini"
DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_VLM_TEMPERATURE = 0.7
DEFAULT_VLM_MAX_NEW_TOKENS = 256
DEFAULT_QUERY_METHOD = "global"
DEFAULT_RESPONSE_TYPE = "Single Short Answer"

_LITELLM_VLM_CACHE: dict[tuple[str, str, str | None, int], Any] = {}
_LOCAL_QWEN_RUNTIME_CACHE: dict[str, tuple[Any, Any]] = {}

VLM_SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi ngắn dựa trên ảnh.

Quy tắc:
- Trả lời bằng tiếng Việt.
- Chỉ dùng thông tin nhìn thấy trong ảnh.
- Nếu câu hỏi hỏi nội dung biển, tên địa điểm, thương hiệu, sản phẩm hoặc một cụm OCR ngắn, hãy trả lại đúng bề mặt chữ khi có thể.
- Không over-generate. Nếu ảnh có thêm chữ thừa nhưng chỉ có một đáp án chính, chỉ trả đáp án chính.
- Với thương hiệu tiếng Anh, giữ nguyên tiếng Anh.
- Nếu ảnh không đủ để trả lời, trả đúng: Không đủ thông tin
- Chỉ trả lời final answer, không giải thích.

EXAMPLES:
Example 1:
Question: Biển ghi gì?
Answer: Cá cắn câu

Example 2:
Question: Nội dung chính là gì?
Answer: Cá cắn câu

Example 3:
Question: Tên thương hiệu là gì?
Answer: Highlands Coffee

Example 4:
Question: Người trong ảnh tên gì?
Answer: Không đủ thông tin
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer a question with GraphRAG if a per-image BYOG workspace exists, otherwise fall back to a VLM."
    )
    parser.add_argument("--image-id", required=True, help="Target image id, e.g. 10061")
    parser.add_argument("question", help="Question to answer for the image.")
    parser.add_argument(
        "--mode",
        choices=["auto", "graphrag", "vlm"],
        default="auto",
        help="Inference mode. Default: auto",
    )
    parser.add_argument(
        "--byog-root",
        type=Path,
        default=DEFAULT_BYOG_ROOT,
        help=f"Root containing per-image BYOG workspaces. Default: {DEFAULT_BYOG_ROOT}",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DEFAULT_IMAGES_ROOT,
        help=f"Root containing image files. Default: {DEFAULT_IMAGES_ROOT}",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=f".env file used to resolve API keys for litellm_api fallback. Default: {DEFAULT_ENV_FILE}",
    )
    parser.add_argument(
        "--vlm-backend",
        choices=["local_qwen", "litellm_api"],
        default=DEFAULT_VLM_BACKEND,
        help=f"VLM backend used for fallback. Default: {DEFAULT_VLM_BACKEND}",
    )
    parser.add_argument(
        "--vlm-provider",
        default=DEFAULT_VLM_PROVIDER,
        help=f"LiteLLM provider used when --vlm-backend=litellm_api. Default: {DEFAULT_VLM_PROVIDER}",
    )
    parser.add_argument(
        "--vlm-api-base",
        default=None,
        help="Optional LiteLLM api_base for OpenAI-compatible / proxy endpoints.",
    )
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable holding the API key when --vlm-backend=litellm_api.",
    )
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_VLM_MODEL,
        help=f"VLM model used for fallback. Default: {DEFAULT_VLM_MODEL}",
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
        default=DEFAULT_VLM_TEMPERATURE,
        help=f"Sampling temperature for VLM fallback. Default: {DEFAULT_VLM_TEMPERATURE}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_VLM_MAX_NEW_TOKENS,
        help=f"Maximum new tokens for local_qwen fallback. Default: {DEFAULT_VLM_MAX_NEW_TOKENS}",
    )
    parser.add_argument(
        "--community-level",
        type=int,
        default=2,
        help="GraphRAG community level. Default: 2",
    )
    parser.add_argument(
        "--response-type",
        default=DEFAULT_RESPONSE_TYPE,
        help=f"GraphRAG response type. Default: {DEFAULT_RESPONSE_TYPE}",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional JSON file to write the answer payload.",
    )
    return parser.parse_args()


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_api_key(env_file: Path, env_name: str) -> str:
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    file_values = parse_env_file(env_file)
    env_value = file_values.get(env_name)
    if env_value:
        os.environ[env_name] = env_value
        return env_value
    raise SystemExit(f"Missing API key. Expected {env_name} in environment or {env_file}")


def resolve_image_path(images_root: Path, image_id: str) -> Path:
    candidates = [
        images_root / f"{image_id}.jpg",
        images_root / f"{image_id}.jpeg",
        images_root / f"{image_id}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Image file not found for image_id={image_id} under {images_root}")


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_workspace(byog_root: Path, image_id: str) -> Path:
    return (byog_root / image_id).resolve()


def has_queryable_workspace(workspace_root: Path) -> bool:
    required = [
        workspace_root / "settings.yaml",
        workspace_root / "output" / "entities.parquet",
        workspace_root / "output" / "communities.parquet",
        workspace_root / "output" / "community_reports.parquet",
    ]
    return all(path.exists() for path in required)


async def run_graphrag_query(
    *,
    workspace_root: Path,
    question: str,
    community_level: int,
    response_type: str,
) -> str:
    try:
        import pandas as pd
        from graphrag import api as graphrag_api
        from graphrag.config.load_config import load_config
    except Exception:
        return await asyncio.to_thread(
            run_graphrag_query_cli,
            workspace_root=workspace_root,
            question=question,
            community_level=community_level,
            response_type=response_type,
        )

    config = load_config(root_dir=workspace_root)
    output_dir = workspace_root / "output"
    entities = pd.read_parquet(output_dir / "entities.parquet")
    communities = pd.read_parquet(output_dir / "communities.parquet")
    community_reports = pd.read_parquet(output_dir / "community_reports.parquet")
    response, _context = await graphrag_api.global_search(
        config=config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=community_level,
        dynamic_community_selection=False,
        response_type=response_type,
        query=question,
        verbose=False,
    )
    return str(response).strip()


def run_graphrag_query_cli(
    *,
    workspace_root: Path,
    question: str,
    community_level: int,
    response_type: str,
) -> str:
    repo_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if not repo_python.exists():
        raise FileNotFoundError(f"GraphRAG runtime not found: {repo_python}")

    command = [
        str(repo_python),
        "-m",
        "graphrag",
        "query",
        question,
        "--root",
        str(workspace_root),
        "--method",
        "global",
        "--community-level",
        str(community_level),
        "--response-type",
        response_type,
    ]
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        message = stderr or stdout or f"GraphRAG CLI query failed with exit code {result.returncode}"
        raise RuntimeError(message)

    output = result.stdout.strip()
    if not output:
        raise RuntimeError("GraphRAG CLI query returned empty output.")
    return output.splitlines()[-1].strip()


def build_vlm_prompt(question: str) -> str:
    return f"{VLM_SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"


def get_litellm_vlm_model(
    *,
    provider: str,
    api_key: str,
    model_name: str,
    api_base: str | None,
    max_retries: int,
):
    cache_key = (provider, model_name, api_base, max_retries)
    cached = _LITELLM_VLM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from graphrag_llm.completion import create_completion
    from graphrag_llm.config.model_config import ModelConfig
    from graphrag_llm.config.types import AuthMethod

    model_config = ModelConfig(
        model_provider=provider,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        auth_method=AuthMethod.ApiKey,
        retry={"type": "exponential_backoff", "max_retries": max_retries},
    )
    model = create_completion(model_config)
    _LITELLM_VLM_CACHE[cache_key] = model
    return model


def init_local_qwen_runtime(model_name: str) -> tuple[Any, Any]:
    cached = _LOCAL_QWEN_RUNTIME_CACHE.get(model_name)
    if cached is not None:
        return cached

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "Missing local Qwen dependencies. Install torch, transformers, and accelerate into the Python env running this script."
        ) from exc

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    runtime = (model, processor)
    _LOCAL_QWEN_RUNTIME_CACHE[model_name] = runtime
    return runtime


def _generate_with_local_qwen(
    *,
    model_name: str,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    model, processor = init_local_qwen_runtime(model_name)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_vlm_prompt(question)},
                {"type": "image", "image": str(image_path.resolve())},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = temperature > 0
    generation_config.temperature = temperature if temperature > 0 else 1.0

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return str(output_text).strip()


async def run_vlm_query(
    *,
    image_path: Path,
    question: str,
    backend: str,
    model_name: str,
    max_retries: int,
    temperature: float = DEFAULT_VLM_TEMPERATURE,
    max_new_tokens: int = DEFAULT_VLM_MAX_NEW_TOKENS,
    provider: str = DEFAULT_VLM_PROVIDER,
    api_base: str | None = None,
    env_file: Path | None = None,
    api_key_env: str | None = None,
) -> str:
    if backend == "local_qwen":
        return await asyncio.to_thread(
            _generate_with_local_qwen,
            model_name=model_name,
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    if env_file is None or api_key_env is None:
        raise ValueError("env_file and api_key_env are required for litellm_api fallback.")

    from graphrag_llm.utils import CompletionContentPartBuilder, CompletionMessagesBuilder

    api_key = resolve_api_key(env_file.resolve(), api_key_env)
    model = get_litellm_vlm_model(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        api_base=api_base,
        max_retries=max_retries,
    )
    content = (
        CompletionContentPartBuilder()
        .add_text_part(f"Câu hỏi: {question}\nTrả lời ngắn gọn theo đúng quy tắc.")
        .add_image_part(image_to_data_url(image_path), "high")
        .build()
    )
    messages = (
        CompletionMessagesBuilder()
        .add_system_message(VLM_SYSTEM_PROMPT)
        .add_user_message(content)
        .build()
    )
    response = await model.completion_async(messages=messages, temperature=0)
    return response.content.strip()


async def main_async() -> None:
    args = parse_args()
    image_id = str(args.image_id).strip()
    byog_root = args.byog_root.resolve()
    images_root = args.images_root.resolve()
    workspace_root = resolve_workspace(byog_root, image_id)
    image_path = resolve_image_path(images_root, image_id)

    used_method = ""
    answer = ""
    fallback_reason: str | None = None

    if args.mode in {"auto", "graphrag"} and has_queryable_workspace(workspace_root):
        try:
            answer = await run_graphrag_query(
                workspace_root=workspace_root,
                question=args.question,
                community_level=args.community_level,
                response_type=args.response_type,
            )
            used_method = "graphrag"
        except Exception as exc:  # pragma: no cover - runtime fallback
            if args.mode == "graphrag":
                raise
            fallback_reason = f"GraphRAG query failed: {type(exc).__name__}: {exc}"

    elif args.mode == "graphrag":
        raise SystemExit(f"Queryable BYOG workspace not found for image {image_id}: {workspace_root}")
    else:
        if not workspace_root.exists():
            fallback_reason = "Per-image BYOG workspace not found"
        elif not has_queryable_workspace(workspace_root):
            fallback_reason = "Per-image BYOG workspace exists but is not indexed/queryable yet"

    if not answer:
        answer = await run_vlm_query(
            image_path=image_path,
            question=args.question,
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

    payload = {
        "image_id": image_id,
        "question": args.question,
        "mode": args.mode,
        "used_method": used_method,
        "answer": answer,
        "image_path": str(image_path),
        "workspace_root": str(workspace_root),
        "fallback_reason": fallback_reason,
        "vlm_backend": args.vlm_backend,
        "vlm_model": args.vlm_model,
    }

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
