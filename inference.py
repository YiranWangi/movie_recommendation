"""Print recommendations using the same engine as the API."""

import argparse
import json
from pathlib import Path

from recommender import MovieRecommender, validate_request
from utils import DEFAULT_DATA, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user_id", type=int, default=1)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        validate_request(args.user_id, args.top_n)
        engine = MovieRecommender(args.model_path, args.data_dir)
        try:
            recommendations = engine.recommend(args.user_id, args.top_n)
        finally:
            engine.close()
        text = json.dumps(recommendations, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
