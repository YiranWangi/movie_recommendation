"""Local Flask service using the shared recommendation engine."""

import argparse
from pathlib import Path

from recommender import MovieRecommender, UnknownUserError, validate_request
from utils import DEFAULT_DATA, DEFAULT_MODEL


def create_app(engine=None, model_path=DEFAULT_MODEL, data_dir=DEFAULT_DATA):
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    if engine is None:
        engine = MovieRecommender(model_path, data_dir)
    app.config["RECOMMENDER"] = engine

    @app.get("/health")
    def health():
        return jsonify({"status": "ready"})

    @app.get("/recommend")
    def recommend():
        try:
            user_id = int(request.args["user_id"])
            top_n = int(request.args.get("top_n", 10))
            validate_request(user_id, top_n)
        except (KeyError, ValueError):
            return jsonify({"error": "Provide a positive user_id and top_n between 1 and 100."}), 400
        try:
            return jsonify(engine.recommend(user_id, top_n))
        except UnknownUserError as error:
            return jsonify({"error": str(error)}), 404

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    engine = MovieRecommender(args.model_path, args.data_dir)
    try:
        create_app(engine=engine).run(host=args.host, port=args.port, debug=False, use_reloader=False)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
