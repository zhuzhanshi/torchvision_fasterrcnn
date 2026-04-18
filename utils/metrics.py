from __future__ import annotations

import csv
import os
from typing import Dict, List

from .file_io import dump_json


def save_eval_outputs(metrics: Dict, per_class_ap: List[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    dump_json(metrics, os.path.join(out_dir, "metrics.json"))
    if per_class_ap:
        csv_path = os.path.join(out_dir, "per_class_ap.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_class_ap[0].keys()))
            writer.writeheader()
            for r in per_class_ap:
                writer.writerow(r)
