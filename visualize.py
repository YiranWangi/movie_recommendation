"""Save validation RMSE for the parameter grid from a completed run."""

import argparse
import json
from pathlib import Path

from utils import DEFAULT_RUN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_RUN / "metrics.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "validation_rmse.png")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    candidates = report["validation_grid"]
    labels = [f"rank={row['parameters']['rank']}\nreg={row['parameters']['regParam']}"
              for row in candidates]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, [row["validation"]["als_rmse"] for row in candidates], color="#426a8c")
    ax.set(ylabel="Validation RMSE (scorable rows)", xlabel="ALS parameters")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
