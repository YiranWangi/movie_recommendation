"""Read the tab/pipe-delimited MovieLens 100K files without changing them."""

import csv
from pathlib import Path


def read_ratings(path):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for line, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(fields) != 4:
                raise ValueError(f"Rating row {line} needs four tab-separated fields.")
            user, movie, rating, timestamp = map(int, fields)
            if not (0 < user < 2**31 and 0 < movie < 2**31 and 1 <= rating <= 5 and timestamp >= 0):
                raise ValueError(f"Invalid values in rating row {line}.")
            if (user, movie) in seen:
                raise ValueError(f"Duplicate user/movie rating at row {line}.")
            seen.add((user, movie))
            rows.append((user, movie, float(rating), timestamp))
    if not rows:
        raise ValueError("Ratings file is empty.")
    return rows


def read_genres(genre_path):
    genre_map = {}
    with Path(genre_path).open(encoding="latin-1", newline="") as handle:
        for fields in csv.reader(handle, delimiter="|"):
            if fields:
                genre_map[int(fields[1])] = fields[0]
    if set(genre_map) != set(range(19)):
        raise ValueError("MovieLens 100K needs genre IDs 0 through 18.")
    return genre_map


def read_movies(path, genre_path):
    genre_map = read_genres(genre_path)
    rows, seen = [], set()
    with Path(path).open(encoding="latin-1", newline="") as handle:
        for line, fields in enumerate(csv.reader(handle, delimiter="|"), 1):
            if len(fields) != 24:
                raise ValueError(f"Movie row {line} needs 24 pipe-separated fields.")
            movie = int(fields[0])
            flags = [int(value) for value in fields[5:]]
            if movie <= 0 or movie in seen or not fields[1] or any(flag not in (0, 1) for flag in flags):
                raise ValueError(f"Invalid movie metadata at row {line}.")
            seen.add(movie)
            genres = [genre_map[index] for index, flag in enumerate(flags) if flag]
            rows.append((movie, fields[1], genres))
    if not rows:
        raise ValueError("Movie metadata is empty.")
    return rows
