from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.file_io import dump_json, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser("Export predictions from infer/eval outputs to csv/txt/json")
    parser.add_argument(
        "--input",
        required=True,
        help="Input predictions path: file (predictions_all.json / predictions.json / per-image infer json) or directory containing json files.",
    )
    parser.add_argument("--output", required=True, help="Output file path.")
    parser.add_argument("--format", default="csv", choices=["csv", "txt", "json"])
    return parser.parse_args()


def _load_any_predictions(input_path: str):
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    rows = []
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "results" in data:  # infer/predictions_all.json
            for item in data["results"]:
                for box, label, score in zip(item.get("boxes", []), item.get("labels", []), item.get("scores", [])):
                    rows.append(
                        {
                            "file_path": item.get("file_path", ""),
                            "file_name": item.get("file_name", ""),
                            "image_id": item.get("relative_id", ""),
                            "category_id": int(label),
                            "score": float(score),
                            "bbox": box,
                        }
                    )
        elif isinstance(data, list) and data and isinstance(data[0], dict) and "image_id" in data[0]:  # eval/predictions.json
            for item in data:
                rows.append(
                    {
                        "file_path": "",
                        "file_name": "",
                        "image_id": item.get("image_id", ""),
                        "category_id": int(item.get("category_id", -1)),
                        "score": float(item.get("score", 0.0)),
                        "bbox": item.get("bbox", []),
                    }
                )
        elif isinstance(data, dict) and "boxes" in data and "labels" in data and "scores" in data:  # infer per-image json
            for box, label, score in zip(data["boxes"], data["labels"], data["scores"]):
                rows.append(
                    {
                        "file_path": data.get("file_path", ""),
                        "file_name": data.get("file_name", ""),
                        "image_id": data.get("relative_id", ""),
                        "category_id": int(label),
                        "score": float(score),
                        "bbox": box,
                    }
                )
        else:
            raise ValueError(f"Unsupported predictions json structure: {input_path}")
    else:
        json_files = sorted(glob.glob(os.path.join(str(p), "*.json")))
        if not json_files:
            raise RuntimeError(f"No json files found under directory: {input_path}")
        for jf in json_files:
            rows.extend(_load_any_predictions(jf))
    return rows


def _write_csv(rows, output_path):
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "file_name", "image_id", "category_id", "score", "bbox"])
        writer.writeheader()
        for r in rows:
            rr = dict(r)
            rr["bbox"] = " ".join(str(x) for x in rr["bbox"])
            writer.writerow(rr)


def _write_txt(rows, output_path):
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            box = " ".join(str(x) for x in r["bbox"])
            f.write(f"{r['image_id']} {r['category_id']} {r['score']:.6f} {box}\n")


def main():
    args = parse_args()
    rows = _load_any_predictions(args.input)
    if args.format == "json":
        dump_json(rows, args.output)
    elif args.format == "csv":
        _write_csv(rows, args.output)
    else:
        _write_txt(rows, args.output)
    print(f"Exported {len(rows)} predictions to: {args.output}")


if __name__ == "__main__":
    main()
