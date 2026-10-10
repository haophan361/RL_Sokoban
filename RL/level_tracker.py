import csv
from datetime import datetime
from pathlib import Path


class LevelTracker:

    def __init__(self, dataset, csv_path):
        self.csv_path = Path(csv_path)
        self.dataset = dataset
        self.total_levels = len(dataset)
        self.solved = {}
        self._dirty = False

        if self.csv_path.exists():
            self._load_csv()

    def _load_csv(self):
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["is_success"]):
                    self.solved[(row["group"], row["level"])] = {
                        "best_pushes": int(row["best_pushes"]),
                        "updated_at": row["updated_at"],
                    }

    def update(self, group, level, solved, pushes):
        if not solved:
            return
        key = (group, level)
        if key not in self.solved:
            self.solved[key] = {
                "best_pushes": pushes,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._dirty = True
        elif pushes < self.solved[key]["best_pushes"]:
            self.solved[key]["best_pushes"] = pushes
            self.solved[key]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._dirty = True

    def flush(self):
        if not self._dirty:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["group", "level", "is_success", "best_pushes", "updated_at"]
            )
            writer.writeheader()
            for idx in range(self.total_levels):
                group, level = self.dataset.get_level_name(idx)
                key = (group, level)
                if key in self.solved:
                    info = self.solved[key]
                    writer.writerow({
                        "group": group, "level": level,
                        "is_success": 1,
                        "best_pushes": info["best_pushes"],
                        "updated_at": info["updated_at"],
                    })
                else:
                    writer.writerow({
                        "group": group, "level": level,
                        "is_success": 0, "best_pushes": 0, "updated_at": "",
                    })
        self._dirty = False

    def summary(self):
        solved = len(self.solved)
        return {
            "total": self.total_levels,
            "solved": solved,
            "unsolved": self.total_levels - solved,
            "solve_rate": solved / self.total_levels if self.total_levels else 0.0,
        }
