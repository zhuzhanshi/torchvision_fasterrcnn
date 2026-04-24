from __future__ import annotations

import csv
import os
from typing import Dict, List

from .file_io import dump_json


def save_eval_outputs(metrics: Dict, per_class_ap: List[dict], out_dir: str, per_class_summary: List[dict] | None = None):
    os.makedirs(out_dir, exist_ok=True)
    dump_json(metrics, os.path.join(out_dir, "metrics.json"))
    if per_class_summary:
        dump_json(per_class_summary, os.path.join(out_dir, "per_class_summary.json"))
        summary_csv = os.path.join(out_dir, "per_class_summary.csv")
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_class_summary[0].keys()))
            writer.writeheader()
            for r in per_class_summary:
                writer.writerow(r)
    if per_class_ap:
        dump_json(per_class_ap, os.path.join(out_dir, "per_class_ap.json"))
        csv_path = os.path.join(out_dir, "per_class_ap.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_class_ap[0].keys()))
            writer.writeheader()
            for r in per_class_ap:
                writer.writerow(r)
