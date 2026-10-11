"""Regression tests for MovieLens parsing and HTTP input handling."""

import tempfile
import unittest
from pathlib import Path

from api import create_app
from data_io import read_movies, read_ratings
from recommender import UnknownUserError, validate_request
from utils import DEFAULT_DATA, DEFAULT_MODEL, DEFAULT_RUN


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.genres = self.root / "u.genre"
        self.genres.write_text("\n".join(f"genre{index}|{index}" for index in range(19)) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_movie_pipe_separator_and_latin1(self):
        fields = ["1", "Café (1995)", "01-Jan-1995", "", "http://example.invalid"]
        fields += ["1" if index in (1, 3) else "0" for index in range(19)]
        path = self.root / "u.item"
        path.write_bytes(("|".join(fields) + "\n").encode("latin-1"))
        self.assertEqual(read_movies(path, self.genres), [(1, "Café (1995)", ["genre1", "genre3"])])

    def test_wrong_movie_delimiter_fails(self):
        path = self.root / "u.item"
        path.write_text("1\tTitle\tDate\n")
        with self.assertRaisesRegex(ValueError, "pipe-separated"):
            read_movies(path, self.genres)

    def test_ratings_keep_timestamp(self):
        path = self.root / "u.data"
        path.write_text("1\t2\t4\t900000000\n")
        self.assertEqual(read_ratings(path), [(1, 2, 4.0, 900000000)])

    def test_invalid_rating_and_duplicate_pair_fail(self):
        for content in ("1\t2\t6\t10\n", "1\t2\t4\t10\n1\t2\t3\t11\n"):
            with self.subTest(content=content):
                path = self.root / "u.data"
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_ratings(path)

    def test_actual_dataset_counts_and_metadata(self):
        ratings = read_ratings(DEFAULT_DATA / "u.data")
        movies = read_movies(DEFAULT_DATA / "u.item", DEFAULT_DATA / "u.genre")
        self.assertEqual(len(ratings), 100000)
        self.assertEqual(len({row[0] for row in ratings}), 943)
        self.assertEqual(len(movies), 1682)
        self.assertTrue({row[1] for row in ratings}.issubset({row[0] for row in movies}))

    def test_shared_default_model_path(self):
        self.assertEqual(DEFAULT_MODEL, DEFAULT_RUN / "model")


class RequestTests(unittest.TestCase):
    def test_positive_bounded_request(self):
        validate_request(1, 10)
        for user, count in ((0, 10), (-1, 10), (True, 10), (1, 0), (1, 101), (2**31, 1)):
            with self.subTest(user=user, count=count):
                with self.assertRaises(ValueError):
                    validate_request(user, count)


class APITests(unittest.TestCase):
    def setUp(self):
        class Engine:
            def __init__(self):
                self.calls = []

            def recommend(self, user, count):
                self.calls.append((user, count))
                if user == 999:
                    raise UnknownUserError("Unknown user.")
                return [{"movieId": 7, "title": "Example", "genres": ["Comedy"], "predicted_rating": 4.1}]
        self.engine = Engine()
        self.client = create_app(engine=self.engine).test_client()

    def test_healthy_service(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ready"})

    def test_valid_request_uses_shared_engine(self):
        response = self.client.get("/recommend?user_id=1&top_n=3")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.engine.calls, [(1, 3)])
        self.assertEqual(response.json[0]["movieId"], 7)

    def test_bad_request_never_calls_engine(self):
        for query in ("", "?user_id=x", "?user_id=1&top_n=0", "?user_id=-1", "?user_id=1&top_n=101"):
            with self.subTest(query=query):
                response = self.client.get("/recommend" + query)
                self.assertEqual(response.status_code, 400)
        self.assertFalse(self.engine.calls)

    def test_unknown_user_returns_404(self):
        response = self.client.get("/recommend?user_id=999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.json)


if __name__ == "__main__":
    unittest.main()
