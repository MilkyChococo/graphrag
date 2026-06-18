from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_FILE = REPO_ROOT / "test_v1.0.json"
DEFAULT_OCR_ROOT = REPO_ROOT / "spdocvqa_ocr"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "graphs"


@dataclass(frozen=True)
class OcrNode:
    id: str
    kind: str
    text: str
    order: int
    bbox: list[float]
    polygon: list[list[float]]
    confidence: str | None

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return self.x1 + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y1 + self.height / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SP-DocVQA OCR spatial graphs for images used by the test file."
    )
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-horizontal-gap", type=float, default=120.0)
    parser.add_argument("--max-vertical-gap", type=float, default=80.0)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.20)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_console(text: Any) -> str:
    return str(text).encode("ascii", errors="backslashreplace").decode("ascii")


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def image_stem(image_value: str) -> str:
    return Path(str(image_value)).stem


def collect_required_image_ids(test_file: Path) -> list[str]:
    payload = load_json(test_file)
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"Expected a SP-DocVQA 'data' list in {test_file}")
    return sorted({image_stem(row["image"]) for row in rows if row.get("image")})


def polygon_to_bbox(points: list[Any]) -> tuple[list[float], list[list[float]]]:
    polygon: list[list[float]] = []
    for idx in range(0, len(points), 2):
        try:
            polygon.append([float(points[idx]), float(points[idx + 1])])
        except (IndexError, TypeError, ValueError):
            continue
    if not polygon:
        return [0.0, 0.0, 0.0, 0.0], []
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)], polygon


def confidence_of(item: dict[str, Any]) -> str | None:
    value = item.get("confidence")
    return None if value is None else str(value)


def iter_ocr_pages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pages = payload.get("recognitionResults")
    if isinstance(pages, list):
        return [page for page in pages if isinstance(page, dict)]
    pages = payload.get("analyzeResult", {}).get("readResults")
    if isinstance(pages, list):
        return [page for page in pages if isinstance(page, dict)]
    return []


def build_nodes(payload: dict[str, Any], image_id: str) -> tuple[list[OcrNode], list[dict[str, Any]]]:
    nodes: list[OcrNode] = []
    line_word_links: list[dict[str, Any]] = []
    line_order = 0
    word_order = 0

    for page_index, page in enumerate(iter_ocr_pages(payload), start=1):
        for line in page.get("lines", []):
            if not isinstance(line, dict):
                continue
            text = normalize_whitespace(line.get("text", ""))
            bbox, polygon = polygon_to_bbox(line.get("boundingBox") or [])
            if not text or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            line_order += 1
            line_id = f"{image_id}:line:{line_order:04d}"
            nodes.append(
                OcrNode(
                    id=line_id,
                    kind="line",
                    text=text,
                    order=line_order,
                    bbox=bbox,
                    polygon=polygon,
                    confidence=confidence_of(line),
                )
            )

            for word in line.get("words", []):
                if not isinstance(word, dict):
                    continue
                word_text = normalize_whitespace(word.get("text", ""))
                word_bbox, word_polygon = polygon_to_bbox(word.get("boundingBox") or [])
                if not word_text or word_bbox[2] <= word_bbox[0] or word_bbox[3] <= word_bbox[1]:
                    continue
                word_order += 1
                word_id = f"{image_id}:word:{word_order:05d}"
                nodes.append(
                    OcrNode(
                        id=word_id,
                        kind="word",
                        text=word_text,
                        order=word_order,
                        bbox=word_bbox,
                        polygon=word_polygon,
                        confidence=confidence_of(word),
                    )
                )
                line_word_links.append(
                    {
                        "source": line_id,
                        "target": word_id,
                        "type": "LINE_CONTAINS",
                        "weight": 1.0,
                        "page": page_index,
                    }
                )
    return nodes, line_word_links


def interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def overlap_ratio(a: OcrNode, b: OcrNode, axis: str) -> float:
    if axis == "y":
        overlap = interval_overlap(a.y1, a.y2, b.y1, b.y2)
        return overlap / max(1.0, min(a.height, b.height))
    overlap = interval_overlap(a.x1, a.x2, b.x1, b.x2)
    return overlap / max(1.0, min(a.width, b.width))


def build_edges(
    nodes: list[OcrNode],
    line_word_links: list[dict[str, Any]],
    max_horizontal_gap: float,
    max_vertical_gap: float,
    min_overlap_ratio: float,
) -> list[dict[str, Any]]:
    edges = list(line_word_links)
    lines = sorted((node for node in nodes if node.kind == "line"), key=lambda n: (n.y1, n.x1, n.order))
    words = sorted((node for node in nodes if node.kind == "word"), key=lambda n: (n.y1, n.x1, n.order))

    for left, right in zip(lines, lines[1:]):
        edges.append({"source": left.id, "target": right.id, "type": "READS_TO", "weight": 1.0})

    by_line: dict[str, list[OcrNode]] = {}
    for edge in line_word_links:
        if edge["type"] != "LINE_CONTAINS":
            continue
        source = edge["source"]
        target = edge["target"]
        word = next((node for node in words if node.id == target), None)
        if word is not None:
            by_line.setdefault(source, []).append(word)
    for line_id, line_words in by_line.items():
        ordered = sorted(line_words, key=lambda n: (n.x1, n.order))
        for left, right in zip(ordered, ordered[1:]):
            edges.append({"source": left.id, "target": right.id, "type": "NEXT_WORD", "weight": 1.0, "line": line_id})

    for node in lines:
        best_right: tuple[tuple[float, float], OcrNode] | None = None
        best_down: tuple[tuple[float, float], OcrNode] | None = None
        for candidate in lines:
            if candidate.id == node.id:
                continue
            right_gap = candidate.x1 - node.x2
            if (
                candidate.center_x > node.center_x
                and -4.0 <= right_gap <= max_horizontal_gap
                and overlap_ratio(node, candidate, "y") >= min_overlap_ratio
            ):
                score = (max(0.0, right_gap), abs(candidate.center_y - node.center_y))
                if best_right is None or score < best_right[0]:
                    best_right = (score, candidate)

            down_gap = candidate.y1 - node.y2
            if (
                candidate.center_y > node.center_y
                and -4.0 <= down_gap <= max_vertical_gap
                and overlap_ratio(node, candidate, "x") >= min_overlap_ratio
            ):
                score = (max(0.0, down_gap), abs(candidate.center_x - node.center_x))
                if best_down is None or score < best_down[0]:
                    best_down = (score, candidate)

        if best_right is not None:
            _, target = best_right
            edges.append({"source": node.id, "target": target.id, "type": "RIGHT_NEIGHBOR", "weight": 1.0})
        if best_down is not None:
            _, target = best_down
            edges.append({"source": node.id, "target": target.id, "type": "DOWN_NEIGHBOR", "weight": 1.0})

    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge["type"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def build_graph(image_id: str, ocr_file: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(ocr_file)
    nodes, line_word_links = build_nodes(payload, image_id)
    edges = build_edges(
        nodes=nodes,
        line_word_links=line_word_links,
        max_horizontal_gap=args.max_horizontal_gap,
        max_vertical_gap=args.max_vertical_gap,
        min_overlap_ratio=args.min_overlap_ratio,
    )
    pages = iter_ocr_pages(payload)
    page = pages[0] if pages else {}
    return {
        "graph_type": "spdocvqa_ocr_spatial_graph",
        "created_at": now_iso(),
        "image_id": image_id,
        "image_name": f"{image_id}.png",
        "source_files": {"ocr_file": str(ocr_file.resolve())},
        "page": {
            "width": page.get("width"),
            "height": page.get("height"),
            "unit": page.get("unit"),
            "clockwise_orientation": page.get("clockwiseOrientation"),
        },
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind,
                "text": node.text,
                "label": node.text,
                "order": node.order,
                "confidence": node.confidence,
                "bbox": node.bbox,
                "polygon": node.polygon,
                "center": [round(node.center_x, 3), round(node.center_y, 3)],
                "size": [round(node.width, 3), round(node.height, 3)],
            }
            for node in nodes
        ],
        "edges": edges,
    }


def graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    node_counter = Counter(node["kind"] for node in graph["nodes"])
    edge_counter = Counter(edge["type"] for edge in graph["edges"])
    return {
        "created_at": graph["created_at"],
        "image_id": graph["image_id"],
        "node_count": graph["node_count"],
        "edge_count": graph["edge_count"],
        "node_type_counts": dict(sorted(node_counter.items())),
        "edge_type_counts": dict(sorted(edge_counter.items())),
    }


def main() -> None:
    args = parse_args()
    image_ids = collect_required_image_ids(args.test_file.resolve())
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    processed: list[dict[str, Any]] = []
    existing: list[dict[str, Any]] = []
    missing: list[str] = []
    skipped_existing = 0

    for image_id in image_ids:
        output_dir = args.output_root / image_id
        graph_path = output_dir / "graph.json"
        if args.resume and graph_path.exists():
            skipped_existing += 1
            summary_path = output_dir / "summary.json"
            if summary_path.exists():
                existing.append(load_json(summary_path))
            continue

        ocr_file = args.ocr_root / f"{image_id}.json"
        if not ocr_file.exists():
            missing.append(image_id)
            continue

        graph = build_graph(image_id, ocr_file, args)
        summary = graph_summary(graph)
        write_json(graph_path, graph)
        write_json(output_dir / "summary.json", summary)
        processed.append(summary)

    manifest = {
        "created_at": now_iso(),
        "test_file": str(args.test_file.resolve()),
        "ocr_root": str(args.ocr_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "requested_image_count": len(image_ids),
        "available_graph_count": len(existing) + len(processed),
        "processed_count": len(processed),
        "skipped_existing_count": skipped_existing,
        "missing_ocr_count": len(missing),
        "missing_ocr_image_ids": missing,
        "summaries": existing + processed,
    }
    write_json(args.output_root / "build_summary.json", manifest)
    print(
        "Processed "
        f"{len(processed)} image(s); skipped_existing={skipped_existing}; "
        f"missing_ocr={len(missing)}; output={safe_console(args.output_root.resolve())}"
    )


if __name__ == "__main__":
    main()
