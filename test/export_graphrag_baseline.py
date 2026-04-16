from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from graphrag.data_model.schemas import (
    DOCUMENTS_FINAL_COLUMNS,
    ENTITIES_FINAL_COLUMNS,
    RELATIONSHIPS_FINAL_COLUMNS,
    TEXT_UNITS_FINAL_COLUMNS,
)
from graphrag.prompts.index.community_report_text_units import (
    COMMUNITY_REPORT_TEXT_PROMPT,
)
from graphrag.prompts.query.basic_search_system_prompt import BASIC_SEARCH_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_knowledge_system_prompt import (
    GENERAL_KNOWLEDGE_INSTRUCTION,
)
from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_reduce_system_prompt import REDUCE_SYSTEM_PROMPT
from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEMANTIC_ROOT = SCRIPT_DIR / "semantic_graphs"
DEFAULT_WORKSPACE_ROOT = SCRIPT_DIR / "graphrag_baseline"
DEFAULT_ENV_FILE = SCRIPT_DIR.parent / ".env"
PROMPTS_DIRNAME = "prompts"
COMMUNITY_REPORT_TEXT_PROMPT_FILENAME = "community_report_text.txt"
LOCAL_SEARCH_PROMPT_FILENAME = "local_search_system_prompt.txt"
BASIC_SEARCH_PROMPT_FILENAME = "basic_search_system_prompt.txt"
GLOBAL_MAP_PROMPT_FILENAME = "global_search_map_system_prompt.txt"
GLOBAL_REDUCE_PROMPT_FILENAME = "global_search_reduce_system_prompt.txt"
GLOBAL_KNOWLEDGE_PROMPT_FILENAME = "global_search_knowledge_system_prompt.txt"
DEFAULT_VECTOR_SIZE = 1024
DEFAULT_COMPLETION_PROVIDER = "local_hf"
DEFAULT_COMPLETION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_COMPLETION_API_KEY_ENV = "HUGGINGFACE_API_KEY"
DEFAULT_COMPLETION_API_BASE_ENV = "HF_COMPLETION_API_BASE"
DEFAULT_EMBEDDING_PROVIDER = "local_hf"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_API_KEY_ENV = "HUGGINGFACE_API_KEY"
DEFAULT_EMBEDDING_API_BASE_ENV = "HF_EMBEDDING_API_BASE"
DEFAULT_WORKFLOWS = [
    "create_communities",
    "create_community_reports_text",
    "generate_text_embeddings",
]
NO_INFORMATION_ANSWER = "Kh\u00f4ng \u0111\u1ee7 th\u00f4ng tin"


GLOBAL_MAP_PROMPT_VITEXTVQA = (
    MAP_SYSTEM_PROMPT.rstrip()
    + """

---Task-specific instructions---

- Answer in Vietnamese.
- When the question asks for text on a sign, a venue name, a product name, a brand, or another short OCR phrase, preserve the exact surface form from the evidence whenever possible.
- Do not infer details that are not explicitly supported by the evidence.
"""
)


GLOBAL_REDUCE_PROMPT_VITEXTVQA = (
    REDUCE_SYSTEM_PROMPT.rstrip()
    + """

---Task-specific instructions for OCR VQA inference---

- Answer in Vietnamese.
- Prefer the shortest directly supported answer.
- When the question asks for text on a sign, a venue name, a product name, a brand, or another short OCR phrase, copy the exact surface form from the evidence whenever possible.
- Do not over-generate. If the evidence contains extra surrounding words but only one salient answer is needed, return only that salient answer.
- Keep English brand names unchanged.
- If the provided reports do not contain enough evidence, answer exactly: Không đủ thông tin
- For short-answer questions, ignore the usual sectioned report style and return only the final answer text with no heading, bullet list, markdown emphasis, or explanation.

EXAMPLES:
Example 1:
Question: Biển ghi gì?
Evidence: Cá cắn câu
Answer: Cá cắn câu

Example 2:
Question: Nội dung chính là gì?
Evidence: dù lâu vẫn đợi Cá cắn câu
Answer: Cá cắn câu

Example 3:
Question: Tên thương hiệu là gì?
Evidence: Highlands Coffee
Answer: Highlands Coffee

Example 4:
Question: Người trong ảnh tên gì?
Evidence: Hello world
Answer: Không đủ thông tin
"""
)


def _append_task_instructions(base_prompt: str, extra: str) -> str:
    return base_prompt.rstrip() + "\n\n" + extra.strip() + "\n"


COMMUNITY_REPORT_TEXT_PROMPT_VITEXTVQA = _append_task_instructions(
    COMMUNITY_REPORT_TEXT_PROMPT,
    """
---Task-specific instructions for OCR VQA workspaces---

- Preserve short visible strings exactly when they appear in the OCR transcript.
- Focus on directly supported product names, brands, venues, addresses, and other OCR evidence.
- Do not invent people, places, or events that are not grounded in the OCR text units.
""",
)


LOCAL_SEARCH_PROMPT_VITEXTVQA = _append_task_instructions(
    LOCAL_SEARCH_SYSTEM_PROMPT,
    f"""
---Task-specific instructions for OCR VQA inference---

- Answer in Vietnamese.
- Use only the provided data tables. Do not add general knowledge.
- Prefer the shortest directly supported answer.
- When the question asks for visible text, a brand, a shop name, an address, or another short OCR phrase, preserve the exact surface form from the evidence whenever possible.
- If the data tables do not contain enough evidence, answer exactly: {NO_INFORMATION_ANSWER}
- For short-answer questions, return only the final answer text with no heading, bullet list, markdown emphasis, explanation, or citations.
""",
)


BASIC_SEARCH_PROMPT_VITEXTVQA = _append_task_instructions(
    BASIC_SEARCH_SYSTEM_PROMPT,
    f"""
---Task-specific instructions for OCR VQA inference---

- Answer in Vietnamese.
- Use only the provided source table.
- Prefer the shortest directly supported answer.
- When the question asks for visible text, a brand, a shop name, an address, or another short OCR phrase, preserve the exact surface form from the evidence whenever possible.
- If the source table does not contain enough evidence, answer exactly: {NO_INFORMATION_ANSWER}
- For short-answer questions, return only the final answer text with no heading, bullet list, markdown emphasis, explanation, or citations.
""",
)


GLOBAL_MAP_PROMPT_VITEXTVQA = _append_task_instructions(
    MAP_SYSTEM_PROMPT,
    """
---Task-specific instructions---

- Answer in Vietnamese.
- Preserve exact visible strings when the evidence clearly contains them.
- Do not infer details that are not explicitly supported by the evidence.
""",
)


GLOBAL_REDUCE_PROMPT_VITEXTVQA = _append_task_instructions(
    REDUCE_SYSTEM_PROMPT,
    f"""
---Task-specific instructions for OCR VQA inference---

- Answer in Vietnamese.
- Prefer the shortest directly supported answer.
- When the question asks for visible text, a brand, a shop name, an address, or another short OCR phrase, preserve the exact surface form from the evidence whenever possible.
- Do not over-generate. If the evidence contains extra surrounding words but only one salient answer is needed, return only that salient answer.
- Keep English brand names unchanged.
- If the provided reports do not contain enough evidence, answer exactly: {NO_INFORMATION_ANSWER}
- For short-answer questions, return only the final answer text with no heading, bullet list, markdown emphasis, explanation, or citations.
""",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export semantic graph JSON files into a GraphRAG BYOG baseline workspace."
    )
    parser.add_argument(
        "--semantic-root",
        type=Path,
        default=DEFAULT_SEMANTIC_ROOT,
        help=f"Folder containing semantic graph subfolders. Default: {DEFAULT_SEMANTIC_ROOT}",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT,
        help=f"GraphRAG baseline workspace root. Default: {DEFAULT_WORKSPACE_ROOT}",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=f".env file to copy into the workspace. Default: {DEFAULT_ENV_FILE}",
    )
    parser.add_argument(
        "--image-id",
        action="append",
        default=[],
        help="Limit export to specific image ids. Repeat the flag or pass comma-separated ids.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of graph folders to export.",
    )
    parser.add_argument(
        "--clean-workspace",
        action="store_true",
        help="Delete the existing workspace output/log/cache folders before exporting.",
    )
    parser.add_argument(
        "--per-image-workspaces",
        action="store_true",
        help="Export one BYOG workspace per image under --workspace-root/<image_id>.",
    )
    parser.add_argument(
        "--completion-provider",
        default=DEFAULT_COMPLETION_PROVIDER,
        help=f"Completion provider for generated GraphRAG settings. Default: {DEFAULT_COMPLETION_PROVIDER}",
    )
    parser.add_argument(
        "--completion-model",
        default=DEFAULT_COMPLETION_MODEL,
        help=f"Completion model for generated GraphRAG settings. Default: {DEFAULT_COMPLETION_MODEL}",
    )
    parser.add_argument(
        "--completion-api-key-env",
        default=DEFAULT_COMPLETION_API_KEY_ENV,
        help=f"Env var name for the completion API key. Default: {DEFAULT_COMPLETION_API_KEY_ENV}",
    )
    parser.add_argument(
        "--completion-api-base-env",
        default=DEFAULT_COMPLETION_API_BASE_ENV,
        help=f"Env var name for the completion api_base. Default: {DEFAULT_COMPLETION_API_BASE_ENV}",
    )
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        help=f"Embedding provider for generated GraphRAG settings. Default: {DEFAULT_EMBEDDING_PROVIDER}",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model for generated GraphRAG settings. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--embedding-api-key-env",
        default=DEFAULT_EMBEDDING_API_KEY_ENV,
        help=f"Env var name for the embedding API key. Default: {DEFAULT_EMBEDDING_API_KEY_ENV}",
    )
    parser.add_argument(
        "--embedding-api-base-env",
        default=DEFAULT_EMBEDDING_API_BASE_ENV,
        help=f"Env var name for the embedding api_base. Default: {DEFAULT_EMBEDDING_API_BASE_ENV}",
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


def ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_graph_paths(semantic_root: Path, image_ids: list[str], limit: int | None) -> list[Path]:
    requested = set(image_ids)
    paths: list[Path] = []
    for child in sorted(semantic_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if requested and child.name not in requested:
            continue
        graph_path = child / "graph.json"
        if graph_path.exists():
            paths.append(graph_path)
    if limit is not None:
        paths = paths[:limit]
    return paths


def compute_degree_map(relationship_rows: list[dict]) -> dict[str, int]:
    degree_map: dict[str, int] = defaultdict(int)
    seen_pairs: set[tuple[str, str]] = set()
    for row in relationship_rows:
        source = row["source"]
        target = row["target"]
        lo, hi = sorted((source, target))
        if (lo, hi) in seen_pairs:
            continue
        seen_pairs.add((lo, hi))
        degree_map[lo] += 1
        degree_map[hi] += 1
    return dict(degree_map)


def ensure_workspace_dirs(
    workspace_root: Path, clean_workspace: bool
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir = workspace_root / "output"
    input_dir = workspace_root / "input"
    logs_dir = workspace_root / "logs"
    cache_dir = workspace_root / "cache"
    prompts_dir = workspace_root / PROMPTS_DIRNAME

    if clean_workspace:
        for path in (output_dir, input_dir, logs_dir, cache_dir, prompts_dir):
            if path.exists():
                shutil.rmtree(path)

    for path in (workspace_root, output_dir, input_dir, logs_dir, cache_dir, prompts_dir):
        path.mkdir(parents=True, exist_ok=True)

    return output_dir, input_dir, logs_dir, cache_dir, prompts_dir


def build_settings_yaml(
    *,
    completion_provider: str,
    completion_model: str,
    completion_api_key_env: str,
    completion_api_base_env: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_api_key_env: str,
    embedding_api_base_env: str,
) -> str:
    return f"""completion_models:
  default_completion_model:
    type: local_hf
    model_provider: {completion_provider}
    model: {completion_model}
    device_map: auto
    torch_dtype: auto
    trust_remote_code: false
    call_args:
      temperature: 0
    retry:
      type: exponential_backoff

embedding_models:
  default_embedding_model:
    type: local_hf
    model_provider: {embedding_provider}
    model: {embedding_model}
    device: auto
    trust_remote_code: false
    batch_size: 8
    max_length: 1024
    normalize_embeddings: true
    retry:
      type: exponential_backoff

input_storage:
  type: file
  base_dir: "input"

output_storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "logs"

cache:
  type: json
  storage:
    type: file
    base_dir: "cache"

vector_store:
  type: lancedb
  db_uri: output\\lancedb
  vector_size: {DEFAULT_VECTOR_SIZE}

workflows: [{", ".join(DEFAULT_WORKFLOWS)}]

embed_text:
  embedding_model_id: default_embedding_model
  names: [text_unit_text, entity_description]

extract_claims:
  enabled: false

community_reports:
  completion_model_id: default_completion_model
  text_prompt: "prompts/{COMMUNITY_REPORT_TEXT_PROMPT_FILENAME}"
  max_length: 2000
  max_input_length: 8000

local_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/{LOCAL_SEARCH_PROMPT_FILENAME}"
  text_unit_prop: 0.85
  community_prop: 0.05
  top_k_entities: 8
  top_k_relationships: 8

basic_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/{BASIC_SEARCH_PROMPT_FILENAME}"
  k: 3

global_search:
  completion_model_id: default_completion_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"
"""


def write_prompt_files(prompts_dir: Path) -> dict[str, str]:
    prompt_payloads = {
        COMMUNITY_REPORT_TEXT_PROMPT_FILENAME: COMMUNITY_REPORT_TEXT_PROMPT_VITEXTVQA,
        LOCAL_SEARCH_PROMPT_FILENAME: LOCAL_SEARCH_PROMPT_VITEXTVQA,
        BASIC_SEARCH_PROMPT_FILENAME: BASIC_SEARCH_PROMPT_VITEXTVQA,
        GLOBAL_MAP_PROMPT_FILENAME: GLOBAL_MAP_PROMPT_VITEXTVQA,
        GLOBAL_REDUCE_PROMPT_FILENAME: GLOBAL_REDUCE_PROMPT_VITEXTVQA,
        GLOBAL_KNOWLEDGE_PROMPT_FILENAME: GENERAL_KNOWLEDGE_INSTRUCTION,
    }
    written_paths: dict[str, str] = {}
    for filename, content in prompt_payloads.items():
        prompt_path = prompts_dir / filename
        prompt_path.write_text(content.rstrip() + "\n", encoding="utf-8")
        written_paths[filename] = str(prompt_path)
    return written_paths


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


def write_workspace_env(
    *,
    source_env_file: Path,
    target_env_file: Path,
    completion_api_key_env: str,
    completion_api_base_env: str,
    embedding_api_key_env: str,
    embedding_api_base_env: str,
) -> None:
    existing_values = parse_env_file(source_env_file)
    env_lines: list[str] = []
    seen_keys: set[str] = set()

    if source_env_file.exists():
        for raw_line in source_env_file.read_text(encoding="utf-8").splitlines():
            env_lines.append(raw_line)
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _value = line.split("=", 1)
            seen_keys.add(key.strip())

    required_entries = {
        completion_api_key_env: existing_values.get(completion_api_key_env),
        completion_api_base_env: existing_values.get(completion_api_base_env),
        embedding_api_key_env: existing_values.get(embedding_api_key_env),
        embedding_api_base_env: existing_values.get(embedding_api_base_env),
    }
    entries_to_write = {
        key: value
        for key, value in required_entries.items()
        if key not in seen_keys and value
    }

    if entries_to_write:
        if env_lines and env_lines[-1].strip():
            env_lines.append("")
        env_lines.append("# GraphRAG BYOG workspace model configuration")
        for key, value in entries_to_write.items():
            env_lines.append(f'{key}="{value}"')

    target_env_file.write_text("\n".join(env_lines).rstrip() + "\n", encoding="utf-8")


def choose_description(values: list[str]) -> str:
    unique_values = ordered_unique(values)
    if not unique_values:
        return ""
    if len(unique_values) == 1:
        return unique_values[0]
    return " ; ".join(unique_values)


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def estimate_token_count(text: str) -> int:
    return len(text.split())


def build_dataframes(
    graph_paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    entity_accumulator: dict[tuple[str, str], dict] = {}
    relationship_accumulator: dict[tuple[str, str], dict] = {}
    text_unit_accumulator: dict[str, dict[str, object]] = {}
    document_accumulator: dict[str, dict[str, object]] = {}
    processed_image_ids: list[str] = []

    for graph_path in graph_paths:
        payload = load_json(graph_path)
        image_id = str(payload.get("image_id") or graph_path.parent.name)
        image_name = str(payload.get("image_name") or f"{image_id}.jpg")
        document_id = f"document::{image_id}"
        processed_image_ids.append(image_id)

        text_unit_ids_for_document: list[str] = []
        full_document_text_parts: list[str] = []
        for index, text_unit in enumerate(payload.get("text_units", []), start=1):
            text_unit_id = str(text_unit.get("id") or f"{image_id}_chunk_{index:03d}")
            text = str(text_unit.get("text", "")).strip()
            if not text:
                continue
            text_unit_accumulator[text_unit_id] = {
                "id": text_unit_id,
                "text": text,
                "document_id": document_id,
            }
            text_unit_ids_for_document.append(text_unit_id)
            full_document_text_parts.append(text)

        document_accumulator[document_id] = {
            "id": document_id,
            "title": image_name,
            "text_unit_ids": ordered_unique(text_unit_ids_for_document),
            "text": "\n\n".join(full_document_text_parts).strip(),
            "creation_date": str(payload.get("created_at", "")),
            "raw_data": json.dumps(
                {
                    "image_id": image_id,
                    "image_name": image_name,
                    "source_files": payload.get("source_files", {}),
                    "ocr_overview": payload.get("ocr_overview", {}),
                },
                ensure_ascii=False,
            ),
        }

        for entity in payload.get("entities", []):
            key = (str(entity["title"]), str(entity["type"]))
            entry = entity_accumulator.setdefault(
                key,
                {
                    "title": key[0],
                    "type": key[1],
                    "descriptions": [],
                    "text_unit_ids": [],
                    "source_image_ids": [],
                },
            )
            raw_descriptions = entity.get("descriptions_raw") or [entity.get("description", "")]
            entry["descriptions"].extend([str(item) for item in raw_descriptions if item])
            entry["text_unit_ids"].extend([str(item) for item in entity.get("text_unit_ids", [])])
            entry["source_image_ids"].append(image_id)

        for relationship in payload.get("relationships", []):
            key = (str(relationship["source"]), str(relationship["target"]))
            entry = relationship_accumulator.setdefault(
                key,
                {
                    "source": key[0],
                    "target": key[1],
                    "descriptions": [],
                    "text_unit_ids": [],
                    "weight": 0.0,
                    "source_image_ids": [],
                },
            )
            raw_descriptions = relationship.get("descriptions_raw") or [
                relationship.get("description", "")
            ]
            entry["descriptions"].extend([str(item) for item in raw_descriptions if item])
            entry["text_unit_ids"].extend(
                [str(item) for item in relationship.get("text_unit_ids", [])]
            )
            entry["weight"] += float(relationship.get("weight", 0.0) or 0.0)
            entry["source_image_ids"].append(image_id)

    entity_rows: list[dict] = []
    for short_id, entry in enumerate(
        sorted(entity_accumulator.values(), key=lambda row: (row["title"], row["type"])),
        start=1,
    ):
        text_unit_ids = ordered_unique(entry["text_unit_ids"])
        entity_rows.append(
            {
                "id": str(uuid4()),
                "human_readable_id": short_id,
                "title": entry["title"],
                "type": entry["type"],
                "description": choose_description(entry["descriptions"]),
                "text_unit_ids": text_unit_ids,
                "frequency": len(text_unit_ids),
                "degree": 0,
            }
        )

    relationship_rows: list[dict] = []
    for short_id, entry in enumerate(
        sorted(
            relationship_accumulator.values(),
            key=lambda row: (row["source"], row["target"]),
        ),
        start=1,
    ):
        relationship_rows.append(
            {
                "id": str(uuid4()),
                "human_readable_id": short_id,
                "source": entry["source"],
                "target": entry["target"],
                "description": choose_description(entry["descriptions"]),
                "weight": float(entry["weight"]),
                "combined_degree": 0,
                "text_unit_ids": ordered_unique(entry["text_unit_ids"]),
            }
        )

    degree_map = compute_degree_map(relationship_rows)
    for row in entity_rows:
        row["degree"] = int(degree_map.get(row["title"], 0))
    for row in relationship_rows:
        row["combined_degree"] = int(
            degree_map.get(row["source"], 0) + degree_map.get(row["target"], 0)
        )

    text_unit_entity_map: dict[str, list[str]] = defaultdict(list)
    for row in entity_rows:
        for text_unit_id in row["text_unit_ids"]:
            text_unit_entity_map[text_unit_id].append(row["id"])

    text_unit_relationship_map: dict[str, list[str]] = defaultdict(list)
    for row in relationship_rows:
        for text_unit_id in row["text_unit_ids"]:
            text_unit_relationship_map[text_unit_id].append(row["id"])

    text_unit_rows: list[dict] = []
    for short_id, text_unit in enumerate(
        sorted(
            text_unit_accumulator.values(),
            key=lambda row: (str(row["document_id"]), str(row["id"])),
        ),
    ):
        text_unit_id = str(text_unit["id"])
        text = str(text_unit["text"])
        text_unit_rows.append(
            {
                "id": text_unit_id,
                "human_readable_id": short_id,
                "text": text,
                "n_tokens": estimate_token_count(text),
                "document_id": str(text_unit["document_id"]),
                "entity_ids": sorted(text_unit_entity_map.get(text_unit_id, [])),
                "relationship_ids": sorted(
                    text_unit_relationship_map.get(text_unit_id, [])
                ),
                "covariate_ids": [],
            }
        )

    document_rows: list[dict] = []
    for short_id, document in enumerate(
        sorted(document_accumulator.values(), key=lambda row: str(row["id"]))
    ):
        document_rows.append(
            {
                "id": str(document["id"]),
                "human_readable_id": short_id,
                "title": str(document["title"]),
                "text": str(document["text"]),
                "text_unit_ids": list(document["text_unit_ids"]),
                "creation_date": str(document["creation_date"]),
                "raw_data": str(document["raw_data"]),
            }
        )

    entities_df = pd.DataFrame(entity_rows, columns=ENTITIES_FINAL_COLUMNS)
    relationships_df = pd.DataFrame(relationship_rows, columns=RELATIONSHIPS_FINAL_COLUMNS)
    text_units_df = pd.DataFrame(text_unit_rows, columns=TEXT_UNITS_FINAL_COLUMNS)
    documents_df = pd.DataFrame(document_rows, columns=DOCUMENTS_FINAL_COLUMNS)
    return entities_df, relationships_df, text_units_df, documents_df, processed_image_ids


def export_workspace(
    *,
    workspace_root: Path,
    graph_paths: list[Path],
    semantic_root: Path,
    env_file: Path,
    clean_workspace: bool,
    completion_provider: str,
    completion_model: str,
    completion_api_key_env: str,
    completion_api_base_env: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_api_key_env: str,
    embedding_api_base_env: str,
) -> dict[str, object]:
    output_dir, _, _, _, prompts_dir = ensure_workspace_dirs(workspace_root, clean_workspace)
    entities_df, relationships_df, text_units_df, documents_df, processed_image_ids = (
        build_dataframes(graph_paths)
    )
    entities_path = output_dir / "entities.parquet"
    relationships_path = output_dir / "relationships.parquet"
    text_units_path = output_dir / "text_units.parquet"
    documents_path = output_dir / "documents.parquet"
    entities_df.to_parquet(entities_path, index=False)
    relationships_df.to_parquet(relationships_path, index=False)
    text_units_df.to_parquet(text_units_path, index=False)
    documents_df.to_parquet(documents_path, index=False)

    settings_path = workspace_root / "settings.yaml"
    settings_path.write_text(
        build_settings_yaml(
            completion_provider=completion_provider,
            completion_model=completion_model,
            completion_api_key_env=completion_api_key_env,
            completion_api_base_env=completion_api_base_env,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_api_key_env=embedding_api_key_env,
            embedding_api_base_env=embedding_api_base_env,
        ),
        encoding="utf-8",
    )
    prompt_paths = write_prompt_files(prompts_dir)
    write_workspace_env(
        source_env_file=env_file,
        target_env_file=workspace_root / ".env",
        completion_api_key_env=completion_api_key_env,
        completion_api_base_env=completion_api_base_env,
        embedding_api_key_env=embedding_api_key_env,
        embedding_api_base_env=embedding_api_base_env,
    )

    manifest: dict[str, object] = {
        "created_at": now_iso(),
        "semantic_root": str(semantic_root),
        "workspace_root": str(workspace_root),
        "processed_graph_count": len(graph_paths),
        "processed_image_ids": processed_image_ids,
        "entity_count": int(len(entities_df)),
        "relationship_count": int(len(relationships_df)),
        "text_unit_count": int(len(text_units_df)),
        "document_count": int(len(documents_df)),
        "entities_path": str(entities_path),
        "relationships_path": str(relationships_path),
        "text_units_path": str(text_units_path),
        "documents_path": str(documents_path),
        "settings_path": str(settings_path),
        "prompt_paths": prompt_paths,
        "completion_provider": completion_provider,
        "completion_model": completion_model,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
    }
    write_manifest(workspace_root / "export_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    image_ids = expand_image_ids(args.image_id)

    if not args.semantic_root.exists():
        raise SystemExit(f"Semantic root does not exist: {args.semantic_root}")

    graph_paths = collect_graph_paths(args.semantic_root, image_ids, args.limit)
    if not graph_paths:
        raise SystemExit(
            f"No graph.json files found under {args.semantic_root}. "
            "Run test/build_semantic_graph.py first."
        )

    if not args.per_image_workspaces:
        manifest = export_workspace(
            workspace_root=args.workspace_root,
            graph_paths=graph_paths,
            semantic_root=args.semantic_root,
            env_file=args.env_file,
            clean_workspace=args.clean_workspace,
            completion_provider=args.completion_provider,
            completion_model=args.completion_model,
            completion_api_key_env=args.completion_api_key_env,
            completion_api_base_env=args.completion_api_base_env,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            embedding_api_key_env=args.embedding_api_key_env,
            embedding_api_base_env=args.embedding_api_base_env,
        )
        print(
            f"Exported {len(graph_paths)} graph(s) to GraphRAG baseline workspace at "
            f"{args.workspace_root.resolve()}"
        )
        print(
            f"Entities: {manifest['entity_count']}, Relationships: {manifest['relationship_count']}"
        )
        return

    args.workspace_root.mkdir(parents=True, exist_ok=True)
    workspace_manifests: list[dict[str, object]] = []
    for graph_path in graph_paths:
        image_id = graph_path.parent.name
        workspace_root = args.workspace_root / image_id
        workspace_manifest = export_workspace(
            workspace_root=workspace_root,
            graph_paths=[graph_path],
            semantic_root=args.semantic_root,
            env_file=args.env_file,
            clean_workspace=args.clean_workspace,
            completion_provider=args.completion_provider,
            completion_model=args.completion_model,
            completion_api_key_env=args.completion_api_key_env,
            completion_api_base_env=args.completion_api_base_env,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            embedding_api_key_env=args.embedding_api_key_env,
            embedding_api_base_env=args.embedding_api_base_env,
        )
        workspace_manifests.append(workspace_manifest)

    root_manifest = {
        "created_at": now_iso(),
        "semantic_root": str(args.semantic_root),
        "workspace_root": str(args.workspace_root),
        "mode": "per_image_workspaces",
        "workspace_count": len(workspace_manifests),
        "image_ids": [manifest["processed_image_ids"][0] for manifest in workspace_manifests],
        "workspaces": workspace_manifests,
    }
    write_manifest(args.workspace_root / "export_manifest.json", root_manifest)

    print(
        f"Exported {len(graph_paths)} graph(s) to per-image GraphRAG baseline workspaces under "
        f"{args.workspace_root.resolve()}"
    )
    print(f"Workspace count: {len(workspace_manifests)}")


if __name__ == "__main__":
    main()
