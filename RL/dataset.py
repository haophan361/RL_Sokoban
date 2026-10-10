from torch.utils.data import Dataset
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


class SokobanDataset(Dataset):
    def __init__(self, dataset_paths, max_workers: int = None):
        super().__init__()
        self.paths = list(dataset_paths)

        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 4) * 2)

        n = len(dataset_paths)
        print(f"Loading {n:,} levels with {max_workers} threads...", flush=True)

        self.data = [None] * n

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_load_json, path): idx
                for idx, path in enumerate(dataset_paths)
            }

            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                self.data[idx] = future.result()
                done += 1
                if done % 10_000 == 0 or done == n:
                    print(f"  Loaded {done:,}/{n:,} ({done/n*100:.1f}%)", flush=True)

        print(f"Dataset ready: {n:,} levels loaded.", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data["map_matrix"], data.get("dead_squares", [])

    def get_level_name(self, idx):
        p = Path(self.paths[idx])
        return p.parent.name, p.stem