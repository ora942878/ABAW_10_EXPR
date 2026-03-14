from pathlib import Path
from typing import Dict, List, Tuple
"""
from pipeline4_model_trains.utils.utils_read_expr_txt import read_expr_txt
"""
def read_expr_txt(txt_path: Path) -> Tuple[List[int], Dict[int, int]]:
    seq, fmap = [], {}
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            try:
                ints = [int(p) for p in parts]
            except ValueError:
                continue
            if len(ints) >= 2 and (ints[-1] in range(-1, 8)) and (ints[0] > 7):
                fmap[int(ints[0])] = int(ints[-1])
            else:
                seq.append(int(ints[-1]))
    return seq, fmap
