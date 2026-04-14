from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from graphrag.data_model.schemas import ENTITIES_FINAL_COLUMNS, RELATIONSHIPS_FINAL_COLUMNS
from graphrag.prompts.query.global_search_knowledge_system_prompt import (
    GENERAL_KNOWLEDGE_INSTRUCTION,
)
from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_reduce_system_prompt import REDUCE_SYSTEM_PROMPT


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEMANTIC_ROOT = SCRIPT_DIR / "semantic_graphs"
DEFAULT_WORKSPACE_ROOT = SCRIPT_DIR / "graphrag_baseline"
DEFAULT_ENV_FILE = SCRIPT_DIR.parent / ".env"
PROMPTS_DIRNAME = "prompts"
GLOBAL_MAP_PROMPT_FILENAME = "global_search_map_system_prompt.txt"
GLOBAL_REDUCE_PROMPT_FILENAME = "global_search_reduce_system_prompt.txt"
GLOBAL_KNOWLEDGE_PROMPT_FILENAME = "global_search_knowledge_system_prompt.txt"


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


def build_settings_yaml() -> str:
    return """completion_models:
  default_completion_model:
    model_provider: gemini
    model: gemini-2.5-flash-lite
    auth_method: api_key
    api_key: ${GEMINI_API_KEY}
    retry:
      type: exponential_backoff

embedding_models:
  default_embedding_model:
    model_provider: gemini
    model: gemini-embedding-001
    auth_method: api_key
    api_key: ${GEMINI_API_KEY}
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

workflows: [create_communities, create_community_reports]

extract_claims:
  enabled: false

community_reports:
  completion_model_id: default_completion_model
  max_length: 2000
  max_input_length: 8000

global_search:
  completion_model_id: default_completion_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"
"""


def write_prompt_files(prompts_dir: Path) -> dict[str, str]:
    prompt_payloads = {
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


def build_dataframes(
    graph_paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    entity_accumulator: dict[tuple[str, str], dict] = {}
    relationship_accumulator: dict[tuple[str, str], dict] = {}
    processed_image_ids: list[str] = []

    for graph_path in graph_paths:
        payload = load_json(graph_path)
        image_id = str(payload.get("image_id") or graph_path.parent.name)
        processed_image_ids.append(image_id)

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

    entities_df = pd.DataFrame(entity_rows, columns=ENTITIES_FINAL_COLUMNS)
    relationships_df = pd.DataFrame(relationship_rows, columns=RELATIONSHIPS_FINAL_COLUMNS)
    return entities_df, relationships_df, processed_image_ids


def export_workspace(
    *,
    workspace_root: Path,
    graph_paths: list[Path],
    semantic_root: Path,
    env_file: Path,
    clean_workspace: bool,
) -> dict[str, object]:
    output_dir, _, _, _, prompts_dir = ensure_workspace_dirs(workspace_root, clean_workspace)
    entities_df, relationships_df, processed_image_ids = build_dataframes(graph_paths)
    entities_path = output_dir / "entities.parquet"
    relationships_path = output_dir / "relationships.parquet"
    entities_df.to_parquet(entities_path, index=False)
    relationships_df.to_parquet(relationships_path, index=False)

    settings_path = workspace_root / "settings.yaml"
    settings_path.write_text(build_settings_yaml(), encoding="utf-8")
    prompt_paths = write_prompt_files(prompts_dir)

    if env_file.exists():
        shutil.copyfile(env_file, workspace_root / ".env")

    manifest: dict[str, object] = {
        "created_at": now_iso(),
        "semantic_root": str(semantic_root),
        "workspace_root": str(workspace_root),
        "processed_graph_count": len(graph_paths),
        "processed_image_ids": processed_image_ids,
        "entity_count": int(len(entities_df)),
        "relationship_count": int(len(relationships_df)),
        "entities_path": str(entities_path),
        "relationships_path": str(relationships_path),
        "settings_path": str(settings_path),
        "prompt_paths": prompt_paths,
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
