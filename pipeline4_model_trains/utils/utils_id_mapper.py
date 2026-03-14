import csv
import os
from pathlib import Path
from typing import Dict

from configs.paths import PATH
"""
from pipeline4_model_trains.utils.utils_id_mapper import IDMapper
"""

class IDMapper:
    def __init__(self, csv_path: Path = PATH.EXPR_VIDEO_INDEX_CSV):
        self.csv_path = csv_path
        self._mapping = self._load_mapping()

    def _load_mapping(self) -> Dict[str, str]:
        mapping = {}
        if not self.csv_path.exists():
            print(f"[IDMapper][WARN]cannot find: {self.csv_path}")
            return mapping

        try:
            with open(self.csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tid = row.get("txtid") or row.get("txt_id") or row.get("sample") or row.get("name")
                    vid = row.get("videoid") or row.get("video_id")

                    if tid and vid:
                        mapping[tid.strip()] = vid.strip()
        except Exception as e:
            print(f"[IDMapper][ERROR] 读取 CSV 失败: {e}")

        return mapping

    def get_videoid(self, txtid: str) -> str:
        return self._mapping.get(txtid, txtid)

    def __len__(self):
        return len(self._mapping)

