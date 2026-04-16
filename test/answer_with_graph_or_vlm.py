from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import mimetypes
import os
import re
import subprocess
import sys
from collections import OrderedDict
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
DEFAULT_SEMANTIC_ROOT = SCRIPT_DIR / "semantic_graphs"
DEFAULT_GRAPHRAG_BACKEND = "graphrag_api"
DEFAULT_VLM_BACKEND = "local_qwen"
DEFAULT_VLM_PROVIDER = "gemini"
DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_GRAPHRAG_COMPLETION_MODEL = DEFAULT_VLM_MODEL
DEFAULT_GRAPHRAG_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_VLM_TEMPERATURE = 0.7
DEFAULT_VLM_MAX_NEW_TOKENS = 256
DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_EMBEDDING_DEVICE = "auto"
DEFAULT_QUERY_METHOD = "local"
DEFAULT_FALLBACK_QUERY_METHOD = "basic"
DEFAULT_RESPONSE_TYPE = "Single Short Answer"
NO_INFORMATION_ANSWER = "Kh\u00f4ng \u0111\u1ee7 th\u00f4ng tin"
MAX_RETRIEVAL_ITEM_CHARS = 1600
MAX_RETRIEVAL_CACHE_SIZE = 256
MIN_GRAPHRAG_API_PYTHON = (3, 11)

_LITELLM_VLM_CACHE: dict[tuple[str, str, str | None, int], Any] = {}
_LOCAL_QWEN_RUNTIME_CACHE: dict[str, tuple[Any, Any]] = {}
_LOCAL_EMBED_RUNTIME_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}
_LOCAL_RETRIEVAL_INDEX_CACHE: OrderedDict[
    tuple[str, str, str, str, str, str],
    dict[str, Any],
] = OrderedDict()
_DATA_REF_PATTERN = re.compile(r"\s*\[Data:[^\]]*\]")
_ANSWER_PREFIX_PATTERN = re.compile(
    r"^(?:answer|final answer|tra loi|tra loi cuoi cung|dap an)\s*[:\-]\s*",
    re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

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

LOCAL_HF_GRAPHRAG_SYSTEM_PROMPT = f"""Ban la tro ly OCR VQA ket hop anh goc va ngu canh GraphRAG cho dung mot anh.

Quy tac:
- Tra loi bang tieng Viet.
- Chi dung thong tin co trong anh hoac trong ngu canh truy hoi.
- Uu tien chuoi OCR nguyen van va noi dung nhin thay truc tiep trong anh.
- Neu graph context mau thuan voi anh hoac OCR transcript, uu tien anh va OCR transcript.
- Khi cau hoi hoi ve bien hieu, thuong hieu, san pham, dia chi, so dien thoai, gia tien hoac cum tu ngan, giu nguyen be mat chu neu co the.
- Neu khong du thong tin, tra loi dung: {NO_INFORMATION_ANSWER}
- Chi tra loi final answer, khong giai thich.
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
        "--semantic-root",
        type=Path,
        default=DEFAULT_SEMANTIC_ROOT,
        help=f"Root containing per-image semantic graph JSON files. Default: {DEFAULT_SEMANTIC_ROOT}",
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
        "--query-method",
        choices=["basic", "local", "global", "drift"],
        default=DEFAULT_QUERY_METHOD,
        help=f"GraphRAG query method. Default: {DEFAULT_QUERY_METHOD}",
    )
    parser.add_argument(
        "--fallback-query-method",
        choices=["none", "basic", "local", "global", "drift"],
        default=DEFAULT_FALLBACK_QUERY_METHOD,
        help=f"Optional GraphRAG fallback query method. Default: {DEFAULT_FALLBACK_QUERY_METHOD}",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Do not auto-build missing GraphRAG query artifacts before querying.",
    )
    parser.add_argument(
        "--graphrag-backend",
        choices=["local_hf", "graphrag_api"],
        default=DEFAULT_GRAPHRAG_BACKEND,
        help=f"GraphRAG inference backend. Default: {DEFAULT_GRAPHRAG_BACKEND}",
    )
    parser.add_argument(
        "--graphrag-completion-model",
        default=DEFAULT_GRAPHRAG_COMPLETION_MODEL,
        help=f"Local Hugging Face generation model used when --graphrag-backend=local_hf. Default: {DEFAULT_GRAPHRAG_COMPLETION_MODEL}",
    )
    parser.add_argument(
        "--graphrag-embedding-model",
        default=DEFAULT_GRAPHRAG_EMBEDDING_MODEL,
        help=f"Local Hugging Face embedding model used when --graphrag-backend=local_hf. Default: {DEFAULT_GRAPHRAG_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
        help=f"Number of retrieved graph/OCR context items to feed into local_hf GraphRAG. Default: {DEFAULT_RETRIEVAL_TOP_K}",
    )
    parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_EMBEDDING_DEVICE,
        help=f"Device for local embedding model. Default: {DEFAULT_EMBEDDING_DEVICE}",
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


def resolve_semantic_graph_path(semantic_root: Path, image_id: str) -> Path:
    return (semantic_root / image_id / "graph.json").resolve()


def normalize_short_answer(answer: str) -> str:
    cleaned = _DATA_REF_PATTERN.sub("", str(answer or "").strip())
    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().strip("*").strip('"').strip("'")
        line = re.sub(r"^[#>\-\*\s]+", "", line)
        line = _ANSWER_PREFIX_PATTERN.sub("", line).strip()
        if line:
            lines.append(line)

    if not lines:
        return cleaned.strip()
    return lines[0].strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [_stringify_value(part) for part in value]
        return ", ".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _truncate_text(text: str, max_chars: int = MAX_RETRIEVAL_ITEM_CHARS) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _vector_store_ready(workspace_root: Path) -> bool:
    lancedb_dir = workspace_root / "output" / "lancedb"
    return lancedb_dir.exists() and any(lancedb_dir.iterdir())


def _required_workspace_paths(workspace_root: Path, query_method: str) -> list[Path]:
    required = [workspace_root / "settings.yaml"]
    output_dir = workspace_root / "output"

    if query_method == "basic":
        required.extend([output_dir / "text_units.parquet"])
    elif query_method == "local":
        required.extend(
            [
                output_dir / "entities.parquet",
                output_dir / "relationships.parquet",
                output_dir / "text_units.parquet",
                output_dir / "communities.parquet",
                output_dir / "community_reports.parquet",
            ]
        )
    elif query_method == "drift":
        required.extend(
            [
                output_dir / "entities.parquet",
                output_dir / "relationships.parquet",
                output_dir / "text_units.parquet",
                output_dir / "communities.parquet",
                output_dir / "community_reports.parquet",
            ]
        )
    else:
        required.extend(
            [
                output_dir / "entities.parquet",
                output_dir / "communities.parquet",
                output_dir / "community_reports.parquet",
            ]
        )
    return required


def has_queryable_workspace(workspace_root: Path, query_method: str = DEFAULT_QUERY_METHOD) -> bool:
    required = _required_workspace_paths(workspace_root, query_method)
    if not all(path.exists() for path in required):
        return False
    if query_method in {"basic", "local", "drift"}:
        return _vector_store_ready(workspace_root)
    return True


def _safe_read_parquet_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pandas as pd
    except Exception:
        return []
    return pd.read_parquet(path).to_dict(orient="records")


def _append_retrieval_item(
    items: list[dict[str, str]],
    *,
    kind: str,
    item_id: str,
    text: str,
) -> None:
    payload = _truncate_text(text)
    if not payload:
        return
    items.append({"kind": kind, "id": item_id, "text": payload})


def _workspace_items_for_query_method(
    *,
    workspace_root: Path,
    query_method: str,
) -> list[dict[str, str]]:
    output_dir = workspace_root / "output"
    items: list[dict[str, str]] = []

    for row in _safe_read_parquet_records(output_dir / "text_units.parquet"):
        item_id = _stringify_value(row.get("id") or row.get("human_readable_id"))
        _append_retrieval_item(
            items,
            kind="text_unit",
            item_id=item_id,
            text=_stringify_value(row.get("text")),
        )

    for row in _safe_read_parquet_records(output_dir / "documents.parquet"):
        title = _stringify_value(row.get("title"))
        text = _stringify_value(row.get("text"))
        if not text:
            continue
        item_id = _stringify_value(row.get("id") or row.get("human_readable_id") or title)
        payload = f"Document: {title}\n{text}" if title else text
        _append_retrieval_item(items, kind="document", item_id=item_id, text=payload)

    if query_method in {"local", "global", "drift"}:
        for row in _safe_read_parquet_records(output_dir / "entities.parquet"):
            title = _stringify_value(row.get("title"))
            entity_type = _stringify_value(row.get("type"))
            description = _stringify_value(row.get("description"))
            payload = "\n".join(
                part
                for part in [
                    f"Entity: {title}" if title else "",
                    f"Type: {entity_type}" if entity_type else "",
                    f"Description: {description}" if description else "",
                ]
                if part
            )
            item_id = _stringify_value(row.get("id") or row.get("human_readable_id") or title)
            _append_retrieval_item(items, kind="entity", item_id=item_id, text=payload)

        for row in _safe_read_parquet_records(output_dir / "relationships.parquet"):
            source = _stringify_value(row.get("source"))
            target = _stringify_value(row.get("target"))
            description = _stringify_value(row.get("description"))
            payload = "\n".join(
                part
                for part in [
                    f"Relationship: {source} -> {target}" if source or target else "",
                    f"Description: {description}" if description else "",
                ]
                if part
            )
            item_id = _stringify_value(row.get("id") or row.get("human_readable_id") or f"{source}->{target}")
            _append_retrieval_item(items, kind="relationship", item_id=item_id, text=payload)

        for row in _safe_read_parquet_records(output_dir / "community_reports.parquet"):
            title = _stringify_value(row.get("title"))
            summary = _stringify_value(row.get("summary"))
            full_content = _stringify_value(
                row.get("full_content") or row.get("content") or row.get("report")
            )
            findings = _stringify_value(row.get("findings"))
            payload = "\n".join(
                part
                for part in [
                    f"Community Report: {title}" if title else "",
                    f"Summary: {summary}" if summary else "",
                    full_content,
                    findings,
                ]
                if part
            )
            item_id = _stringify_value(row.get("id") or row.get("community") or title)
            _append_retrieval_item(items, kind="community_report", item_id=item_id, text=payload)

    return items


def _semantic_items_for_query_method(
    *,
    semantic_root: Path,
    image_id: str,
    query_method: str,
) -> list[dict[str, str]]:
    graph_path = resolve_semantic_graph_path(semantic_root, image_id)
    if not graph_path.exists():
        return []

    payload = load_json(graph_path)
    items: list[dict[str, str]] = []

    for text_unit in payload.get("text_units", []):
        _append_retrieval_item(
            items,
            kind="text_unit",
            item_id=_stringify_value(text_unit.get("id")),
            text=_stringify_value(text_unit.get("text")),
        )

    if query_method in {"basic", "local", "drift"}:
        transcript_preview = payload.get("ocr_overview", {}).get("transcript_preview", [])
        if isinstance(transcript_preview, list) and transcript_preview:
            _append_retrieval_item(
                items,
                kind="document",
                item_id=f"{image_id}::ocr_preview",
                text="OCR Preview:\n" + "\n".join(
                    _stringify_value(line) for line in transcript_preview if line
                ),
            )

    if query_method in {"local", "global", "drift"}:
        for entity in payload.get("entities", []):
            title = _stringify_value(entity.get("title"))
            entity_type = _stringify_value(entity.get("type"))
            description = _stringify_value(entity.get("description"))
            payload_text = "\n".join(
                part
                for part in [
                    f"Entity: {title}" if title else "",
                    f"Type: {entity_type}" if entity_type else "",
                    f"Description: {description}" if description else "",
                ]
                if part
            )
            _append_retrieval_item(
                items,
                kind="entity",
                item_id=title or _stringify_value(entity.get("id")),
                text=payload_text,
            )

        for relationship in payload.get("relationships", []):
            source = _stringify_value(relationship.get("source"))
            target = _stringify_value(relationship.get("target"))
            description = _stringify_value(relationship.get("description"))
            payload_text = "\n".join(
                part
                for part in [
                    f"Relationship: {source} -> {target}" if source or target else "",
                    f"Description: {description}" if description else "",
                ]
                if part
            )
            _append_retrieval_item(
                items,
                kind="relationship",
                item_id=f"{source}->{target}",
                text=payload_text,
            )

    return items


def load_local_hf_corpus(
    *,
    workspace_root: Path,
    semantic_root: Path,
    image_id: str,
    query_method: str,
) -> list[dict[str, str]]:
    combined_items: list[dict[str, str]] = []
    if workspace_root.exists():
        combined_items.extend(
            _workspace_items_for_query_method(
                workspace_root=workspace_root,
                query_method=query_method,
            )
        )
    combined_items.extend(
        _semantic_items_for_query_method(
            semantic_root=semantic_root,
            image_id=image_id,
            query_method=query_method,
        )
    )

    deduped: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in combined_items:
        dedupe_key = (item["kind"], item["text"])
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped.append(item)
    return deduped


def _resolve_embedding_device(device_preference: str) -> str:
    if device_preference in {"cpu", "cuda"}:
        return device_preference
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_local_embedding_runtime(
    *,
    model_name: str,
    device_preference: str,
) -> tuple[Any, Any, str]:
    cache_key = (model_name, device_preference)
    cached = _LOCAL_EMBED_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "Missing local embedding dependencies. Install torch and transformers into the Python env running this script."
        ) from exc

    device = _resolve_embedding_device(device_preference)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    runtime = (model, tokenizer, device)
    _LOCAL_EMBED_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _embed_texts_with_local_hf(
    *,
    model_name: str,
    texts: list[str],
    device_preference: str,
    batch_size: int = 8,
) -> Any:
    import torch

    if not texts:
        return torch.empty((0, 0), dtype=torch.float32)

    model, tokenizer, device = init_local_embedding_runtime(
        model_name=model_name,
        device_preference=device_preference,
    )
    all_embeddings: list[Any] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.no_grad():
            outputs = model(**encoded)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            last_hidden_state = outputs[0]
        pooled = _mean_pool(last_hidden_state, encoded["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(normalized.cpu())

    return torch.cat(all_embeddings, dim=0)


def _retrieval_kind_bonus(query_method: str, kind: str) -> float:
    bonus_table = {
        "basic": {
            "text_unit": 0.20,
            "document": 0.15,
            "entity": 0.03,
            "relationship": 0.01,
            "community_report": 0.00,
        },
        "local": {
            "text_unit": 0.16,
            "document": 0.10,
            "entity": 0.05,
            "relationship": 0.03,
            "community_report": 0.02,
        },
        "global": {
            "text_unit": 0.04,
            "document": 0.05,
            "entity": 0.08,
            "relationship": 0.06,
            "community_report": 0.12,
        },
        "drift": {
            "text_unit": 0.16,
            "document": 0.10,
            "entity": 0.05,
            "relationship": 0.03,
            "community_report": 0.02,
        },
    }
    return bonus_table.get(query_method, bonus_table["local"]).get(kind, 0.0)


def _tokenize_words(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_PATTERN.findall(str(text or ""))
        if len(token.strip()) > 1
    }


def _lexical_overlap_bonus(question_tokens: set[str], candidate_text: str) -> float:
    if not question_tokens:
        return 0.0
    overlap = len(question_tokens & _tokenize_words(candidate_text))
    if overlap <= 0:
        return 0.0
    return min(0.18, overlap * 0.03)


def _cache_local_retrieval_index(
    cache_key: tuple[str, str, str, str, str, str],
    payload: dict[str, Any],
) -> None:
    _LOCAL_RETRIEVAL_INDEX_CACHE[cache_key] = payload
    _LOCAL_RETRIEVAL_INDEX_CACHE.move_to_end(cache_key)
    while len(_LOCAL_RETRIEVAL_INDEX_CACHE) > MAX_RETRIEVAL_CACHE_SIZE:
        _LOCAL_RETRIEVAL_INDEX_CACHE.popitem(last=False)


def _format_query_for_embedding(question: str) -> str:
    return f"Represent this sentence for searching relevant passages: {question.strip()}"


def retrieve_local_hf_context(
    *,
    image_id: str,
    workspace_root: Path,
    semantic_root: Path,
    question: str,
    query_method: str,
    embedding_model_name: str,
    retrieval_top_k: int,
    embedding_device: str,
) -> list[dict[str, Any]]:
    import torch

    cache_key = (
        image_id,
        str(workspace_root.resolve()),
        str(semantic_root.resolve()),
        query_method,
        embedding_model_name,
        embedding_device,
    )
    cached = _LOCAL_RETRIEVAL_INDEX_CACHE.get(cache_key)
    if cached is None:
        items = load_local_hf_corpus(
            workspace_root=workspace_root,
            semantic_root=semantic_root,
            image_id=image_id,
            query_method=query_method,
        )
        if not items:
            raise FileNotFoundError(
                f"No local GraphRAG corpus found for image {image_id} under {workspace_root} or {semantic_root}"
            )
        embeddings = _embed_texts_with_local_hf(
            model_name=embedding_model_name,
            texts=[item["text"] for item in items],
            device_preference=embedding_device,
        )
        cached = {"items": items, "embeddings": embeddings}
        _cache_local_retrieval_index(cache_key, cached)

    question_embedding = _embed_texts_with_local_hf(
        model_name=embedding_model_name,
        texts=[_format_query_for_embedding(question)],
        device_preference=embedding_device,
    )[0]
    similarity_scores = torch.mv(cached["embeddings"], question_embedding.cpu())
    question_tokens = _tokenize_words(question)

    ranked: list[dict[str, Any]] = []
    for index, item in enumerate(cached["items"]):
        base_score = float(similarity_scores[index].item())
        adjusted_score = (
            base_score
            + _retrieval_kind_bonus(query_method, item["kind"])
            + _lexical_overlap_bonus(question_tokens, item["text"])
        )
        ranked.append(
            {
                **item,
                "base_score": base_score,
                "score": adjusted_score,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[: max(1, retrieval_top_k)]


def build_local_hf_graphrag_prompt(
    *,
    question: str,
    retrieved_items: list[dict[str, Any]],
) -> str:
    context_blocks: list[str] = []
    for rank, item in enumerate(retrieved_items, start=1):
        context_blocks.append(
            f"[Context {rank}] {item['kind']} | score={item['score']:.3f}\n{item['text']}"
        )

    retrieved_context = "\n\n".join(context_blocks).strip()
    return (
        f"{LOCAL_HF_GRAPHRAG_SYSTEM_PROMPT}\n\n"
        f"Retrieved context:\n{retrieved_context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


async def run_graphrag_query(
    *,
    workspace_root: Path,
    question: str,
    query_method: str,
    community_level: int,
    response_type: str,
) -> str:
    if sys.version_info < MIN_GRAPHRAG_API_PYTHON:
        return await asyncio.to_thread(
            run_graphrag_query_cli,
            workspace_root=workspace_root,
            question=question,
            query_method=query_method,
            community_level=community_level,
            response_type=response_type,
        )

    try:
        import pandas as pd
        from graphrag import api as graphrag_api
        from graphrag.config.load_config import load_config
    except Exception:
        return await asyncio.to_thread(
            run_graphrag_query_cli,
            workspace_root=workspace_root,
            question=question,
            query_method=query_method,
            community_level=community_level,
            response_type=response_type,
        )

    config = load_config(root_dir=workspace_root)
    output_dir = workspace_root / "output"
    if query_method == "basic":
        text_units = pd.read_parquet(output_dir / "text_units.parquet")
        response, _context = await graphrag_api.basic_search(
            config=config,
            text_units=text_units,
            response_type=response_type,
            query=question,
            verbose=False,
        )
        return normalize_short_answer(str(response))

    if query_method == "local":
        entities = pd.read_parquet(output_dir / "entities.parquet")
        communities = pd.read_parquet(output_dir / "communities.parquet")
        community_reports = pd.read_parquet(output_dir / "community_reports.parquet")
        text_units = pd.read_parquet(output_dir / "text_units.parquet")
        relationships = pd.read_parquet(output_dir / "relationships.parquet")
        covariates = (
            pd.read_parquet(output_dir / "covariates.parquet")
            if (output_dir / "covariates.parquet").exists()
            else None
        )
        response, _context = await graphrag_api.local_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            covariates=covariates,
            community_level=community_level,
            response_type=response_type,
            query=question,
            verbose=False,
        )
        return normalize_short_answer(str(response))

    if query_method == "drift":
        entities = pd.read_parquet(output_dir / "entities.parquet")
        communities = pd.read_parquet(output_dir / "communities.parquet")
        community_reports = pd.read_parquet(output_dir / "community_reports.parquet")
        text_units = pd.read_parquet(output_dir / "text_units.parquet")
        relationships = pd.read_parquet(output_dir / "relationships.parquet")
        response, _context = await graphrag_api.drift_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            community_level=community_level,
            response_type=response_type,
            query=question,
            verbose=False,
        )
        return normalize_short_answer(str(response))

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
    return normalize_short_answer(str(response))


def run_graphrag_query_cli(
    *,
    workspace_root: Path,
    question: str,
    query_method: str,
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
        query_method,
        "--response-type",
        response_type,
    ]
    if query_method != "basic":
        command.extend(["--community-level", str(community_level)])
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        message = stderr or stdout or f"GraphRAG CLI query failed with exit code {result.returncode}"
        raise RuntimeError(message)

    output = (result.stdout or "").strip()
    if not output:
        raise RuntimeError("GraphRAG CLI query returned empty output.")
    return normalize_short_answer(output.splitlines()[-1].strip())


def run_graphrag_index_cli(*, workspace_root: Path) -> None:
    repo_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if not repo_python.exists():
        raise FileNotFoundError(f"GraphRAG runtime not found: {repo_python}")

    command = [
        str(repo_python),
        "-m",
        "graphrag",
        "index",
        "--root",
        str(workspace_root),
        "--skip-validation",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        message = stderr or stdout or f"GraphRAG CLI index failed with exit code {result.returncode}"
        raise RuntimeError(message)


async def ensure_graphrag_index(
    *,
    workspace_root: Path,
    query_method: str,
) -> bool:
    if has_queryable_workspace(workspace_root, query_method):
        return False
    if not workspace_root.exists():
        raise FileNotFoundError(f"Per-image BYOG workspace not found: {workspace_root}")

    if sys.version_info < MIN_GRAPHRAG_API_PYTHON:
        await asyncio.to_thread(run_graphrag_index_cli, workspace_root=workspace_root)
        if not has_queryable_workspace(workspace_root, query_method):
            raise RuntimeError(
                f"Workspace is still not queryable after indexing: {workspace_root}"
            )
        return True

    try:
        from graphrag import api as graphrag_api
        from graphrag.config.load_config import load_config
    except Exception:
        await asyncio.to_thread(run_graphrag_index_cli, workspace_root=workspace_root)
    else:
        config = load_config(root_dir=workspace_root)
        outputs = await graphrag_api.build_index(config=config, verbose=False)
        errors = [output for output in outputs if output.error is not None]
        if errors:
            message = "; ".join(
                f"{output.workflow}: {output.error}" for output in errors[:3]
            )
            raise RuntimeError(f"GraphRAG index failed for {workspace_root}: {message}")

    if not has_queryable_workspace(workspace_root, query_method):
        raise RuntimeError(
            f"Workspace is still not queryable after indexing: {workspace_root}"
        )
    return True


async def answer_with_graphrag(
    *,
    image_id: str,
    question: str,
    byog_root: Path,
    semantic_root: Path,
    images_root: Path,
    query_method: str,
    fallback_query_method: str | None,
    community_level: int,
    response_type: str,
    auto_index: bool,
    backend: str,
    completion_model_name: str,
    embedding_model_name: str,
    retrieval_top_k: int,
    temperature: float,
    max_new_tokens: int,
    embedding_device: str,
) -> dict[str, Any]:
    workspace_root = resolve_workspace(byog_root.resolve(), image_id)
    fallback_reason: str | None = None
    index_built = False

    methods_to_try: list[str] = []
    for method in [query_method, fallback_query_method]:
        if method is None:
            continue
        if method in methods_to_try:
            continue
        methods_to_try.append(method)

    for method in methods_to_try:
        try:
            if backend == "local_hf":
                image_path = resolve_image_path(images_root, image_id)
                retrieved_items = await asyncio.to_thread(
                    retrieve_local_hf_context,
                    image_id=image_id,
                    workspace_root=workspace_root,
                    semantic_root=semantic_root.resolve(),
                    question=question,
                    query_method=method,
                    embedding_model_name=embedding_model_name,
                    retrieval_top_k=retrieval_top_k,
                    embedding_device=embedding_device,
                )
                prompt_override = build_local_hf_graphrag_prompt(
                    question=question,
                    retrieved_items=retrieved_items,
                )
                answer = await asyncio.to_thread(
                    _generate_with_local_qwen,
                    model_name=completion_model_name,
                    image_path=image_path,
                    question=question,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    prompt_override=prompt_override,
                )
            else:
                if auto_index:
                    index_built = (
                        await ensure_graphrag_index(
                            workspace_root=workspace_root,
                            query_method=method,
                        )
                        or index_built
                    )
                elif not has_queryable_workspace(workspace_root, method):
                    raise RuntimeError(
                        f"Queryable BYOG workspace not found for image {image_id} using method {method}: {workspace_root}"
                    )

                answer = await run_graphrag_query(
                    workspace_root=workspace_root,
                    question=question,
                    query_method=method,
                    community_level=community_level,
                    response_type=response_type,
                )
            return {
                "image_id": image_id,
                "question": question,
                "answer": normalize_short_answer(answer),
                "used_method": "graphrag",
                "query_method_used": method,
                "workspace_root": str(workspace_root),
                "fallback_reason": fallback_reason,
                "index_built": index_built,
                "graphrag_backend": backend,
            }
        except Exception as exc:
            if method == fallback_query_method:
                raise
            fallback_reason = (
                f"Primary GraphRAG query failed with {backend}/{query_method}: "
                f"{type(exc).__name__}: {exc}"
            )

    raise RuntimeError(f"GraphRAG could not answer image {image_id}")


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
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    runtime = (model, processor)
    _LOCAL_QWEN_RUNTIME_CACHE[model_name] = runtime
    return runtime


def _prepare_local_qwen_inputs(
    *,
    processor: Any,
    image_path: Path,
    prompt_text: str,
) -> Any:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def _generate_with_local_qwen(
    *,
    model_name: str,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    temperature: float,
    prompt_override: str | None = None,
) -> str:
    import torch

    model, processor = init_local_qwen_runtime(model_name)
    prompt_text = prompt_override or build_vlm_prompt(question)
    inputs = _prepare_local_qwen_inputs(
        processor=processor,
        image_path=image_path,
        prompt_text=prompt_text,
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

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
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
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
        answer = await asyncio.to_thread(
            _generate_with_local_qwen,
            model_name=model_name,
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return normalize_short_answer(answer)

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
    return normalize_short_answer(response.content.strip())


async def main_async() -> None:
    args = parse_args()
    image_id = str(args.image_id).strip()
    byog_root = args.byog_root.resolve()
    images_root = args.images_root.resolve()
    workspace_root = resolve_workspace(byog_root, image_id)
    image_path = resolve_image_path(images_root, image_id)
    fallback_query_method = (
        None if args.fallback_query_method == "none" else args.fallback_query_method
    )

    used_method = ""
    answer = ""
    fallback_reason: str | None = None
    query_method_used: str | None = None
    index_built = False

    if args.mode in {"auto", "graphrag"}:
        try:
            graphrag_payload = await answer_with_graphrag(
                image_id=image_id,
                question=args.question,
                byog_root=byog_root,
                semantic_root=args.semantic_root.resolve(),
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
            query_method_used = graphrag_payload["query_method_used"]
            fallback_reason = graphrag_payload.get("fallback_reason")
            index_built = bool(graphrag_payload.get("index_built"))
        except Exception as exc:  # pragma: no cover - runtime fallback
            if args.mode == "graphrag":
                raise
            fallback_reason = f"GraphRAG query failed: {type(exc).__name__}: {exc}"
    else:
        if not workspace_root.exists():
            fallback_reason = "Per-image BYOG workspace not found"
        elif not has_queryable_workspace(workspace_root, args.query_method):
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
        "query_method_requested": args.query_method,
        "query_method_used": query_method_used,
        "answer": normalize_short_answer(answer),
        "image_path": str(image_path),
        "workspace_root": str(workspace_root),
        "fallback_reason": fallback_reason,
        "index_built": index_built,
        "graphrag_backend": args.graphrag_backend,
        "graphrag_completion_model": args.graphrag_completion_model,
        "graphrag_embedding_model": args.graphrag_embedding_model,
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
