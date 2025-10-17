# pcd_loader.py
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Iterator
import open3d as o3d
import torch
from torch.utils.data import Dataset

# --- разбор строк "relative/path.pcd 380.600000 ..." ---
_PAT = re.compile(r"""^\s*(?P<path>.+?\.pcd)\s+(?P<bio>[+-]?\d+(?:\.\d+)?)\b""",
                  re.IGNORECASE)

def parse_labels_file(labels_txt: Path) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    with open(labels_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _PAT.match(line)
            if m:
                rel = m.group("path").strip().lstrip("/\\")
                bio = float(m.group("bio"))
                items.append((rel, bio))
    # если дубликаты путей — берём последний
    last: Dict[str, float] = {}
    for rel, bio in items:
        last[rel.replace("\\", "/")] = bio
    return list(last.items())

def read_pcd_points(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(pcd.points, dtype=np.float32)

class PCDBiomassDatasetTorch(Dataset):
    """
    PyTorch-совместимый датасет.
    Возвращает: (pts: (Ni,3) float32 np.ndarray, bio: float, abs_path: str)
    """
    def __init__(self, root: str | Path, labels_txt: str | Path,
                 strict_exists: bool = True, cache_points: bool = False):
        self.root = Path(root).resolve()
        self.pairs = parse_labels_file(Path(labels_txt))
        self.strict_exists = strict_exists
        self.cache_points = cache_points
        self._cache: Dict[str, np.ndarray] = {}

        self._items: List[Tuple[Path, float]] = []
        missing: List[str] = []
        for rel, bio in self.pairs:
            abs_path = (self.root / rel).resolve()
            if not abs_path.exists():
                missing.append(rel)
                if not strict_exists:
                    continue
            else:
                self._items.append((abs_path, bio))
        if strict_exists and missing:
            miss = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} файлов не найдены под root={self.root}.\nПримеры:\n{miss}\n"
                f"Либо поправь пути, либо используй strict_exists=False."
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        abs_path, bio = self._items[idx]
        key = str(abs_path)
        if self.cache_points and key in self._cache:
            pts = self._cache[key]
        else:
            pts = read_pcd_points(abs_path)
            if self.cache_points:
                self._cache[key] = pts
        # возвращаем numpy — collate_fn сам преобразует в тензоры и паддинг
        return pts.astype(np.float32), float(bio), key

    def iter(self) -> Iterator[Tuple[np.ndarray, float, str]]:
        for i in range(len(self)):
            yield self[i]

def collate_points_with_mask(batch):
    """
    batch: list of (pts: (Ni,3) np.float32, bio: float, path: str)
    → points: (B, T_max, 3) float32, mask: (B, T_max) bool, y: (B,) float32
    """
    import numpy as np
    pts_list, y_list, _ = zip(*batch)
    lengths = [p.shape[0] for p in pts_list]
    T_max = max(lengths) if lengths else 0
    B = len(pts_list)

    points = np.zeros((B, T_max, 3), dtype=np.float32)
    mask   = np.zeros((B, T_max), dtype=bool)
    for i, p in enumerate(pts_list):
        n = p.shape[0]
        points[i, :n] = p
        mask[i, :n] = True

    points = torch.from_numpy(points)          # (B, T, 3)
    mask   = torch.from_numpy(mask)            # (B, T)
    y      = torch.tensor(y_list, dtype=torch.float32)  # (B,)
    return (points, mask), y