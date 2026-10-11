"""Load one ALS model and recommend movies the user has not rated."""

import hashlib
import json
from pathlib import Path

from utils import DEFAULT_DATA, DEFAULT_MODEL, get_spark_session, load_movies_u, load_ratings_u


class UnknownUserError(ValueError):
    pass


def validate_request(user_id, top_n):
    if type(user_id) is not int or not 0 < user_id < 2**31:
        raise ValueError("user_id must be a positive 32-bit integer.")
    if type(top_n) is not int or not 1 <= top_n <= 100:
        raise ValueError("top_n must be between 1 and 100.")


class MovieRecommender:
    def __init__(self, model_path=DEFAULT_MODEL, data_dir=DEFAULT_DATA):
        from pyspark.ml.recommendation import ALSModel
        model_path, data_dir = Path(model_path), Path(data_dir)
        if not model_path.is_dir():
            raise ValueError(f"Model directory does not exist: {model_path}. Run train.py first.")
        metadata = model_path.parent / "metrics.json"
        if metadata.exists():
            expected = json.loads(metadata.read_text(encoding="utf-8"))["data_sha256"]
            for name, sha in expected.items():
                if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != sha:
                    raise ValueError(f"Data file differs from the saved training run: {name}")
        self.spark = get_spark_session("MovieLensRecommendations")
        try:
            self.model = ALSModel.load(str(model_path))
            self.movies = load_movies_u(data_dir / "u.item", self.spark).cache()
            self.ratings = load_ratings_u(data_dir / "u.data", self.spark).cache()
        except Exception:
            self.spark.stop()
            raise

    def recommend(self, user_id, top_n=10):
        from pyspark.sql import functions as F
        validate_request(user_id, top_n)
        if not self.model.userFactors.filter(F.col("id") == user_id).limit(1).count():
            raise UnknownUserError("User has no learned factors in this training split.")
        seen = self.ratings.filter(F.col("userId") == user_id).select("movieId").distinct()
        candidates = (self.model.itemFactors.select(F.col("id").alias("movieId"))
                      .join(self.movies, "movieId")
                      .join(seen, "movieId", "left_anti")
                      .withColumn("userId", F.lit(user_id)))
        ranked = (self.model.transform(candidates)
                  .filter(F.col("prediction").isNotNull() & ~F.isnan("prediction"))
                  .orderBy(F.desc("prediction"), F.asc("movieId")).limit(top_n))
        return [{"movieId": row.movieId, "title": row.title,
                 "genres": row.genres, "predicted_rating": float(row.prediction)}
                for row in ranked.select("movieId", "title", "genres", "prediction").collect()]

    def close(self):
        self.spark.stop()
