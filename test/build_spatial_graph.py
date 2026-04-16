from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OCR_ROOT = SCRIPT_DIR / "OCR_img"
DEFAULT_BBOX_ROOT = SCRIPT_DIR / "test_bboxes"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "graphs"


@dataclass(frozen=True)
class Node:
    id: str
    image_id: str
    image_name: str
    text: str
    text_raw: str
    order: int
    det_score: float
    bbox: list[float]
    polygon: list[list[float]]
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    center_x: float
    center_y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a simple spatial graph from OCR outputs."
    )
    parser.add_argument(
        "--ocr-root",
        type=Path,
        default=DEFAULT_OCR_ROOT,
        help=f"Folder containing OCR folders. Default: {DEFAULT_OCR_ROOT}",
    )
    parser.add_argument(
        "--bbox-root",
        type=Path,
        default=DEFAULT_BBOX_ROOT,
        help=f"Folder containing bbox folders. Default: {DEFAULT_BBOX_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Folder where graph outputs will be written. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--image-id",
        action="append",
        default=[],
        help="Image id to process. Repeat the flag or pass comma-separated ids.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of images to process.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Discard OCR rows with det_score below this value.",
    )
    parser.add_argument(
        "--max-horizontal-gap",
        type=float,
        default=80.0,
        help="Maximum gap in pixels for RIGHT_NEIGHBOR edges.",
    )
    parser.add_argument(
        "--max-vertical-gap",
        type=float,
        default=60.0,
        help="Maximum gap in pixels for DOWN_NEIGHBOR edges.",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.25,
        help="Minimum overlap ratio used for spatial neighbors.",
    )
    parser.add_argument(
        "--overlap-iou",
        type=float,
        default=0.05,
        help="IoU threshold for OVERLAP edges.",
    )
    return parser.parse_args()


def expand_image_ids(values: list[str]) -> list[str]:
    image_ids: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                image_ids.append(item)
    return image_ids


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


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


def mojibake_score(text: str) -> int:
    return sum(text.count(token) for token in ("Ã", "Æ", "Ð", "�"))


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
        path = image_dir / filename
        if path.exists():
            return path
    return None


def load_ocr_rows(image_dir: Path) -> list[dict[str, Any]]:
    ocr_file = find_ocr_file(image_dir)
    if ocr_file is None:
        return []
    if ocr_file.suffix == ".jsonl":
        return load_jsonl(ocr_file)
    payload = load_json(ocr_file)
    if isinstance(payload, list):
        return payload
    return []


def parse_bbox(row: dict[str, Any]) -> list[float] | None:
    bbox = row.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(value) for value in bbox]

    polygon = row.get("polygon")
    if isinstance(polygon, list) and polygon:
        xs = [float(point[0]) for point in polygon if isinstance(point, list) and len(point) >= 2]
        ys = [float(point[1]) for point in polygon if isinstance(point, list) and len(point) >= 2]
        if xs and ys:
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def normalize_polygon(row: dict[str, Any], bbox: list[float]) -> list[list[float]]:
    polygon = row.get("polygon")
    if isinstance(polygon, list):
        normalized: list[list[float]] = []
        for point in polygon:
            if isinstance(point, list) and len(point) >= 2:
                normalized.append([float(point[0]), float(point[1])])
        if normalized:
            return normalized
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def rows_to_nodes(rows: list[dict[str, Any]], min_score: float) -> list[Node]:
    nodes: list[Node] = []

    for index, row in enumerate(rows, start=1):
        bbox = parse_bbox(row)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            continue

        det_score = float(row.get("det_score", 0.0) or 0.0)
        if det_score < min_score:
            continue

        image_id = str(row.get("image_id") or row.get("image") or "")
        image_name = str(row.get("image_name") or f"{image_id}.jpg")
        order = int(row.get("order") or row.get("index") or index)
        raw_text = str(row.get("text") or "")
        repaired_text = maybe_fix_mojibake(raw_text)
        text = normalize_whitespace(repaired_text)
        node_id = str(row.get("bbox_id") or f"img_{image_id}_bbox_{index:04d}")

        nodes.append(
            Node(
                id=node_id,
                image_id=image_id,
                image_name=image_name,
                text=text,
                text_raw=raw_text,
                order=order,
                det_score=det_score,
                bbox=[x1, y1, x2, y2],
                polygon=normalize_polygon(row, bbox),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                width=width,
                height=height,
                center_x=x1 + width / 2.0,
                center_y=y1 + height / 2.0,
            )
        )

    return sorted(nodes, key=lambda node: (node.order, node.y1, node.x1, node.id))


def interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def bbox_iou(node_a: Node, node_b: Node) -> float:
    inter_w = interval_overlap(node_a.x1, node_a.x2, node_b.x1, node_b.x2)
    inter_h = interval_overlap(node_a.y1, node_a.y2, node_b.y1, node_b.y2)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    union_area = node_a.width * node_a.height + node_b.width * node_b.height - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def vertical_overlap_ratio(node_a: Node, node_b: Node) -> float:
    overlap = interval_overlap(node_a.y1, node_a.y2, node_b.y1, node_b.y2)
    return overlap / max(1.0, min(node_a.height, node_b.height))


def horizontal_overlap_ratio(node_a: Node, node_b: Node) -> float:
    overlap = interval_overlap(node_a.x1, node_a.x2, node_b.x1, node_b.x2)
    return overlap / max(1.0, min(node_a.width, node_b.width))


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge["type"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def build_edges(
    nodes: list[Node],
    max_horizontal_gap: float,
    max_vertical_gap: float,
    min_overlap_ratio: float,
    overlap_iou: float,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []

    ordered_nodes = sorted(nodes, key=lambda node: (node.order, node.y1, node.x1))
    for left, right in zip(ordered_nodes, ordered_nodes[1:]):
        edges.append(
            {
                "source": left.id,
                "target": right.id,
                "type": "READS_TO",
                "weight": 1.0,
                "source_order": left.order,
                "target_order": right.order,
            }
        )

    for node in nodes:
        best_right: tuple[tuple[float, float, float], Node] | None = None
        best_down: tuple[tuple[float, float, float], Node] | None = None

        for candidate in nodes:
            if candidate.id == node.id:
                continue

            right_gap = candidate.x1 - node.x2
            right_overlap = vertical_overlap_ratio(node, candidate)
            if (
                candidate.center_x > node.center_x
                and -4.0 <= right_gap <= max_horizontal_gap
                and right_overlap >= min_overlap_ratio
            ):
                score = (max(0.0, right_gap), abs(candidate.center_y - node.center_y), -right_overlap)
                if best_right is None or score < best_right[0]:
                    best_right = (score, candidate)

            down_gap = candidate.y1 - node.y2
            down_overlap = horizontal_overlap_ratio(node, candidate)
            if (
                candidate.center_y > node.center_y
                and -4.0 <= down_gap <= max_vertical_gap
                and down_overlap >= min_overlap_ratio
            ):
                score = (max(0.0, down_gap), abs(candidate.center_x - node.center_x), -down_overlap)
                if best_down is None or score < best_down[0]:
                    best_down = (score, candidate)

            iou = bbox_iou(node, candidate)
            if iou >= overlap_iou and node.id < candidate.id:
                edges.append(
                    {
                        "source": node.id,
                        "target": candidate.id,
                        "type": "OVERLAP",
                        "weight": round(iou, 6),
                        "iou": round(iou, 6),
                    }
                )

        if best_right is not None:
            _, target = best_right
            edges.append(
                {
                    "source": node.id,
                    "target": target.id,
                    "type": "RIGHT_NEIGHBOR",
                    "weight": 1.0,
                    "gap": round(max(0.0, target.x1 - node.x2), 3),
                    "overlap_ratio": round(vertical_overlap_ratio(node, target), 6),
                }
            )

        if best_down is not None:
            _, target = best_down
            edges.append(
                {
                    "source": node.id,
                    "target": target.id,
                    "type": "DOWN_NEIGHBOR",
                    "weight": 1.0,
                    "gap": round(max(0.0, target.y1 - node.y2), 3),
                    "overlap_ratio": round(horizontal_overlap_ratio(node, target), 6),
                }
            )

    return dedupe_edges(edges)


def build_graph_payload(
    nodes: list[Node],
    edges: list[dict[str, Any]],
    image_id: str,
    image_name: str,
    ocr_dir: Path,
    bbox_dir: Path,
) -> dict[str, Any]:
    return {
        "graph_type": "ocr_spatial_graph",
        "created_at": now_iso(),
        "image_id": image_id,
        "image_name": image_name,
        "source_files": {
            "ocr_dir": str(ocr_dir),
            "ocr_file": str(find_ocr_file(ocr_dir) or ""),
            "bbox_dir": str(bbox_dir) if bbox_dir.exists() else None,
            "bbox_file": str((bbox_dir / "bboxes.jsonl")) if (bbox_dir / "bboxes.jsonl").exists() else None,
        },
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [
            {
                "id": node.id,
                "text": node.text,
                "text_raw": node.text_raw,
                "label": node.text or "[EMPTY]",
                "order": node.order,
                "det_score": node.det_score,
                "bbox": node.bbox,
                "polygon": node.polygon,
                "center": [round(node.center_x, 3), round(node.center_y, 3)],
                "size": [round(node.width, 3), round(node.height, 3)],
            }
            for node in nodes
        ],
        "edges": edges,
    }


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    edge_counter = Counter(edge["type"] for edge in payload["edges"])
    text_count = sum(1 for node in payload["nodes"] if node["text"])
    return {
        "created_at": payload["created_at"],
        "image_id": payload["image_id"],
        "image_name": payload["image_name"],
        "node_count": payload["node_count"],
        "edge_count": payload["edge_count"],
        "nodes_with_text": text_count,
        "nodes_empty_text": payload["node_count"] - text_count,
        "edge_type_counts": dict(sorted(edge_counter.items())),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def collect_image_ids(ocr_root: Path, requested_ids: list[str], limit: int | None) -> list[str]:
    if requested_ids:
        image_ids = sorted(dict.fromkeys(requested_ids))
    else:
        image_ids = sorted(path.name for path in ocr_root.iterdir() if path.is_dir())
    if limit is not None:
        image_ids = image_ids[:limit]
    return image_ids


def process_image(image_id: str, args: argparse.Namespace) -> dict[str, Any] | None:
    ocr_dir = args.ocr_root / image_id
    bbox_dir = args.bbox_root / image_id
    rows = load_ocr_rows(ocr_dir)
    nodes = rows_to_nodes(rows, min_score=args.min_score)
    if not nodes:
        return None

    image_name = nodes[0].image_name
    edges = build_edges(
        nodes,
        max_horizontal_gap=args.max_horizontal_gap,
        max_vertical_gap=args.max_vertical_gap,
        min_overlap_ratio=args.min_overlap_ratio,
        overlap_iou=args.overlap_iou,
    )
    graph = build_graph_payload(nodes, edges, image_id, image_name, ocr_dir, bbox_dir)
    summary = build_summary(graph)

    output_dir = args.output_root / image_id
    write_json(output_dir / "graph.json", graph)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    image_ids = collect_image_ids(
        ocr_root=args.ocr_root,
        requested_ids=expand_image_ids(args.image_id),
        limit=args.limit,
    )
    if not image_ids:
        raise SystemExit(f"No image ids found under {args.ocr_root}")

    summaries: list[dict[str, Any]] = []
    skipped: list[str] = []

    for image_id in image_ids:
        summary = process_image(image_id, args)
        if summary is None:
            skipped.append(image_id)
            continue
        summaries.append(summary)

    manifest = {
        "created_at": now_iso(),
        "ocr_root": str(args.ocr_root),
        "bbox_root": str(args.bbox_root),
        "output_root": str(args.output_root),
        "processed_count": len(summaries),
        "skipped_count": len(skipped),
        "skipped_image_ids": skipped,
        "summaries": summaries,
    }
    write_json(args.output_root / "build_summary.json", manifest)

    print(
        f"Processed {len(summaries)} image(s). "
        f"Graphs written to {args.output_root.resolve()}"
    )
    if skipped:
        print(f"Skipped {len(skipped)} image(s) with no usable OCR rows.")


if __name__ == "__main__":
    main()
