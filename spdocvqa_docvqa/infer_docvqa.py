from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_FILE = REPO_ROOT / "test_v1.0.json"
DEFAULT_GRAPHS_ROOT = SCRIPT_DIR / "graphs"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "predictions"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "field",
    "for",
    "from",
    "graph",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "page",
    "plotted",
    "shown",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['./-][a-z0-9]+)?", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic OCR-graph baseline for SP-DocVQA test questions."
    )
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--graphs-root", type=Path, default=DEFAULT_GRAPHS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def image_stem(image_value: str) -> str:
    return Path(str(image_value)).stem


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text or ""))]


def content_tokens(text: str) -> list[str]:
    return [token for token in tokens(text) if token not in STOPWORDS and len(token) > 1]


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


def line_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for node in graph.get("nodes", []) if node.get("kind") == "line"]


def is_low_value_line(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return (
        not lowered
        or lowered.startswith("source:")
        or "industrydocuments.ucsf.edu" in lowered
        or lowered in {"continued", "confidential"}
    )


def neighbor_lines(lines: list[dict[str, Any]], index: int, radius: int = 2) -> list[dict[str, Any]]:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return lines[start:end]


def score_line(question_terms: list[str], line: dict[str, Any], position: int) -> float:
    line_terms = tokens(line.get("text", ""))
    if not line_terms:
        return 0.0
    line_set = set(line_terms)
    q_counter = Counter(question_terms)
    overlap = sum(count for term, count in q_counter.items() if term in line_set)
    rare_bonus = sum(1.5 for term in set(question_terms) if len(term) >= 5 and term in line_set)
    phrase_bonus = 0.0
    lowered = str(line.get("text", "")).lower()
    for term in set(question_terms):
        if len(term) >= 4 and term in lowered:
            phrase_bonus += 0.25
    return overlap * 2.0 + rare_bonus + phrase_bonus - position * 0.0001


def clean_answer(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    cleaned = cleaned.strip(" :;-|")
    return cleaned[:300]


def answer_after_label(line_text: str, question_terms: list[str]) -> str | None:
    lower = line_text.lower()
    best_pos: int | None = None
    for term in sorted(set(question_terms), key=len, reverse=True):
        if len(term) < 3:
            continue
        pos = lower.find(term)
        if pos >= 0:
            end = pos + len(term)
            if best_pos is None or end > best_pos:
                best_pos = end
    if best_pos is None:
        return None
    tail = line_text[best_pos:]
    tail = re.sub(r"^[\s:;,\-.)/]+", "", tail)
    if 1 <= len(tail.split()) <= 12:
        return clean_answer(tail)
    return None


def infer_answer(question: str, graph: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    lines = [
        line
        for line in sorted(line_nodes(graph), key=lambda node: (node["bbox"][1], node["bbox"][0], node["order"]))
        if not is_low_value_line(str(line.get("text", "")))
    ]
    q_terms = content_tokens(question)
    if not lines:
        return "", {"reason": "no_lines"}

    scored = [(score_line(q_terms, line, idx), idx, line) for idx, line in enumerate(lines)]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_idx, best_line = scored[0]

    label_answer = answer_after_label(str(best_line.get("text", "")), q_terms)
    if label_answer:
        return label_answer, {
            "reason": "same_line_label_tail",
            "score": best_score,
            "line_id": best_line.get("id"),
            "line_text": best_line.get("text"),
        }

    context = [
        line
        for line in neighbor_lines(lines, best_idx, radius=1)
        if not is_low_value_line(str(line.get("text", "")))
    ]
    answer_line = best_line
    best_text = str(best_line.get("text", ""))
    best_word_count = len(best_text.split())
    label_like = best_text.rstrip().endswith(":") or best_word_count <= 4
    if len(context) > 1 and best_score > 0 and label_like:
        later = [line for line in context if line["bbox"][1] >= best_line["bbox"][1] and line["id"] != best_line["id"]]
        short_later = [line for line in later if 1 <= len(str(line.get("text", "")).split()) <= 8]
        if short_later:
            answer_line = min(short_later, key=lambda line: (abs(line["bbox"][0] - best_line["bbox"][0]), line["bbox"][1]))

    return clean_answer(str(answer_line.get("text", ""))), {
        "reason": "best_graph_line" if answer_line is best_line else "nearby_short_line",
        "score": best_score,
        "line_id": answer_line.get("id"),
        "line_text": answer_line.get("text"),
        "matched_line_id": best_line.get("id"),
        "matched_line_text": best_line.get("text"),
    }


def build_submission(source_payload: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_name": source_payload.get("dataset_name"),
        "dataset_version": source_payload.get("dataset_version"),
        "dataset_split": source_payload.get("dataset_split"),
        "data": [
            {
                "questionId": int(record["questionId"]),
                "answer": record["answer"],
            }
            for record in records
        ],
    }


def main() -> None:
    args = parse_args()
    source_payload = load_json(args.test_file.resolve())
    questions = list(source_payload.get("data", []))
    if args.limit is not None:
        questions = questions[: args.limit]

    output_jsonl = args.output_root / "docvqa_predictions.jsonl"
    output_submission = args.output_root / "docvqa_submission.json"
    output_summary = args.output_root / "docvqa_summary.json"

    if not args.resume and output_jsonl.exists():
        output_jsonl.write_text("", encoding="utf-8")

    completed = load_completed(output_jsonl) if args.resume else {}
    records = list(completed.values())
    missing_graph = 0
    processed = 0
    empty_answers = 0

    graph_cache: dict[str, dict[str, Any]] = {}

    for row in questions:
        question_id = int(row["questionId"])
        if question_id in completed:
            continue

        image_id = image_stem(row["image"])
        graph_path = args.graphs_root / image_id / "graph.json"
        if not graph_path.exists():
            answer = ""
            detail = {"reason": "missing_graph", "graph_path": str(graph_path)}
            missing_graph += 1
        else:
            graph = graph_cache.get(image_id)
            if graph is None:
                graph = load_json(graph_path)
                graph_cache[image_id] = graph
            answer, detail = infer_answer(str(row["question"]), graph)

        if not answer:
            empty_answers += 1

        record = {
            "questionId": question_id,
            "image_id": image_id,
            "image": row.get("image"),
            "question": row.get("question"),
            "answer": answer,
            "method": "ocr_graph_extractive_baseline",
            "detail": detail,
            "created_at": now_iso(),
        }
        append_jsonl(output_jsonl, record)
        records.append(record)
        processed += 1

    records.sort(key=lambda record: int(record["questionId"]))
    write_json(output_submission, build_submission(source_payload, records))
    write_json(
        output_summary,
        {
            "created_at": now_iso(),
            "test_file": str(args.test_file.resolve()),
            "graphs_root": str(args.graphs_root.resolve()),
            "output_jsonl": str(output_jsonl.resolve()),
            "output_submission": str(output_submission.resolve()),
            "requested_question_count": len(questions),
            "processed_count": processed,
            "resumed_count": len(completed),
            "record_count": len(records),
            "missing_graph_count": missing_graph,
            "empty_answer_count": empty_answers,
            "method": "ocr_graph_extractive_baseline",
        },
    )
    print(
        f"Processed {processed} question(s); resumed={len(completed)}; "
        f"missing_graph={missing_graph}; empty_answers={empty_answers}; "
        f"output={safe_console(output_submission.resolve())}"
    )


if __name__ == "__main__":
    main()
