from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from export_graphrag_baseline import export_workspace, write_manifest  # noqa: E402


DEFAULT_TEST_FILE = REPO_ROOT / "test_v1.0.json"
DEFAULT_SEMANTIC_ROOT = SCRIPT_DIR / "semantic_graphs"
DEFAULT_BYOG_ROOT = SCRIPT_DIR / "byog_workspaces"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROVIDER = "local_hf"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SP-DocVQA semantic graphs into per-image BYOG workspaces."
    )
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_BYOG_ROOT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--completion-provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--completion-model", default=DEFAULT_MODEL)
    parser.add_argument("--completion-api-key-env", default="HUGGINGFACE_API_KEY")
    parser.add_argument("--completion-api-base-env", default="HF_COMPLETION_API_BASE")
    parser.add_argument("--embedding-provider", default="local_hf")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-api-key-env", default="HUGGINGFACE_API_KEY")
    parser.add_argument("--embedding-api-base-env", default="HF_EMBEDDING_API_BASE")
    parser.add_argument("--image-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clean-workspace", action="store_true")
    return parser.parse_args()


def rewrite_completion_settings_for_api(
    *,
    workspace_root: Path,
    provider: str,
    model: str,
) -> None:
    settings_path = workspace_root / "settings.yaml"
    if not settings_path.exists():
        return
    text = settings_path.read_text(encoding="utf-8")
    old_block = f"""  default_completion_model:
    type: local_hf
    model_provider: {provider}
    model: {model}
    device_map: auto
    torch_dtype: auto
    trust_remote_code: false
    call_args:
      temperature: 0
    retry:
      type: exponential_backoff"""
    new_block = f"""  default_completion_model:
    type: litellm
    model_provider: {provider}
    model: {model}
    call_args:
      temperature: 0
    retry:
      type: exponential_backoff"""
    if old_block in text:
        settings_path.write_text(text.replace(old_block, new_block), encoding="utf-8")


def expand_image_ids(values: list[str]) -> list[str]:
    image_ids: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                image_ids.append(item)
    return list(dict.fromkeys(image_ids))


def test_image_ids(test_file: Path) -> list[str]:
    import json

    payload = json.loads(test_file.read_text(encoding="utf-8"))
    return sorted({Path(row["image"]).stem for row in payload.get("data", []) if row.get("image")})


def main() -> None:
    args = parse_args()
    requested = expand_image_ids(args.image_id) or test_image_ids(args.test_file)
    if args.limit is not None:
        requested = requested[: args.limit]

    args.workspace_root.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, object]] = []
    missing: list[str] = []
    for image_id in requested:
        graph_path = args.semantic_root / image_id / "graph.json"
        if not graph_path.exists():
            missing.append(image_id)
            continue
        manifest = export_workspace(
            workspace_root=args.workspace_root / image_id,
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
        if args.completion_provider != "local_hf":
            rewrite_completion_settings_for_api(
                workspace_root=args.workspace_root / image_id,
                provider=args.completion_provider,
                model=args.completion_model,
            )
        manifests.append(manifest)

    write_manifest(
        args.workspace_root / "export_manifest.json",
        {
            "mode": "spdocvqa_per_image_byog",
            "test_file": str(args.test_file.resolve()),
            "semantic_root": str(args.semantic_root.resolve()),
            "workspace_root": str(args.workspace_root.resolve()),
            "requested_image_count": len(requested),
            "workspace_count": len(manifests),
            "missing_semantic_count": len(missing),
            "missing_semantic_image_ids": missing,
            "completion_provider": args.completion_provider,
            "completion_model": args.completion_model,
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model,
            "workspaces": manifests,
        },
    )
    print(
        f"Prepared BYOG export under {args.workspace_root.resolve()} "
        f"(workspaces={len(manifests)}, missing_semantic={len(missing)})"
    )


if __name__ == "__main__":
    main()
