"""Exercise a saved model; set MOVIELENS_TEST_MODEL after running train.py."""

import json
import os
import unittest
from pathlib import Path

from api import create_app
from data_io import read_ratings
from recommender import MovieRecommender
from utils import DEFAULT_DATA

MODEL_PATH = os.environ.get("MOVIELENS_TEST_MODEL")


@unittest.skipUnless(MODEL_PATH, "Set MOVIELENS_TEST_MODEL to a saved ALS model directory.")
class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = MovieRecommender(MODEL_PATH, DEFAULT_DATA)
        cls.seen = {movie for user, movie, _, _ in read_ratings(DEFAULT_DATA / "u.data") if user == 1}

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_recommendations_exclude_seen_movies_and_are_sorted(self):
        results = self.engine.recommend(1, 5)
        self.assertEqual(len(results), 5)
        self.assertFalse({item["movieId"] for item in results} & self.seen)
        self.assertEqual(len({item["movieId"] for item in results}), 5)
        self.assertTrue(all(item["title"] and isinstance(item["genres"], list) for item in results))
        scores = [item["predicted_rating"] for item in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_flask_routes_use_real_saved_model(self):
        client = create_app(engine=self.engine).test_client()
        response = client.get("/recommend?user_id=1&top_n=3")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json), 3)
        self.assertFalse({item["movieId"] for item in response.json} & self.seen)
        self.assertEqual(client.get("/recommend?user_id=abc").status_code, 400)
        self.assertEqual(client.get("/recommend?user_id=999999").status_code, 404)

    def test_report_accounts_for_all_test_rows_and_baseline(self):
        report = json.loads((Path(MODEL_PATH).parent / "metrics.json").read_text())
        selected = min(report["validation_grid"], key=lambda row: row["validation"]["als_rmse"])
        self.assertEqual(report["best_parameters"], selected["parameters"])
        self.assertEqual(self.engine.model.rank, report["best_parameters"]["rank"])
        self.assertEqual(sum(report["split_sizes"].values()), report["rating_rows"])
        test = report["test"]
        self.assertEqual(test["total_rows"], report["split_sizes"]["test"])
        self.assertEqual(test["scored_rows"] + test["unscored_rows"], test["total_rows"])
        self.assertEqual(test["coverage"], test["scored_rows"] / test["total_rows"])
        self.assertGreater(test["als_rmse"], 0)
        self.assertGreater(test["global_mean_rmse_on_same_rows"], 0)


if __name__ == "__main__":
    unittest.main()
