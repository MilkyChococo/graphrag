# SP-DocVQA Baseline Pipeline

This folder is a self-contained SP-DocVQA baseline. It only uses images listed in
`../test_v1.0.json` and maps each image path like `documents/rnbx0223_193.png` to
`../spdocvqa_ocr/rnbx0223_193.json`.

Image files are read from the repo root by default, so `documents/rnbx0223_193.png`
is expected at `../documents/rnbx0223_193.png`. Use `--images-root` if the image
folder lives elsewhere.

Default Qwen local Transformers settings:

- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- provider: `local_hf`
- token env for model download: `HUGGINGFACE_API_KEY` or `HF_TOKEN`

Local Qwen-VL dependencies:

```powershell
pip install torch accelerate qwen-vl-utils
pip install git+https://github.com/huggingface/transformers
```

## 1. Spatial OCR Graphs

```powershell
python spdocvqa_docvqa\build_graphs.py --resume
```

Outputs:

- `spdocvqa_docvqa/graphs/<image_id>/graph.json`
- `spdocvqa_docvqa/graphs/build_summary.json`

## 2. Semantic Graphs With Local Qwen-VL Transformers

```powershell
python spdocvqa_docvqa\build_semantic_graphs.py --resume
```

This sends both OCR text and the page image to local Qwen2.5-VL when the image file exists.

Outputs:

- `spdocvqa_docvqa/semantic_graphs/<image_id>/graph.json`
- `spdocvqa_docvqa/semantic_graphs/build_summary.json`

Useful smoke test:

```powershell
python spdocvqa_docvqa\build_semantic_graphs.py --limit 5
```

## 3. BYOG Workspaces

```powershell
python spdocvqa_docvqa\export_byog.py --clean-workspace
```

Outputs:

- `spdocvqa_docvqa/byog_workspaces/<image_id>/`
- `spdocvqa_docvqa/byog_workspaces/export_manifest.json`

## 4. Inference With Local Qwen-VL Transformers

```powershell
python spdocvqa_docvqa\infer_qwen_baseline.py --resume
```

This sends the page image plus semantic/BYOG context to local Qwen2.5-VL.

Outputs:

- `spdocvqa_docvqa/qwen_predictions/docvqa_qwen_predictions.jsonl`
- `spdocvqa_docvqa/qwen_predictions/docvqa_qwen_submission.json`
- `spdocvqa_docvqa/qwen_predictions/docvqa_qwen_summary.json`

The older `infer_docvqa.py` file is a deterministic OCR-only fallback baseline.
The Qwen baseline entrypoint is `infer_qwen_baseline.py`.
