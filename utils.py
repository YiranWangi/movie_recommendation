"""Shared Spark session, loaders and artifact paths."""

import json
from pathlib import Path

from data_io import read_genres, read_movies, read_ratings

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data"
DEFAULT_RUN = ROOT / "models" / "als_run"
DEFAULT_MODEL = DEFAULT_RUN / "model"


def get_spark_session(app_name="MovieRecommender"):
    import sys
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName(app_name).master("local[2]")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.ui.enabled", "false")
             .config("spark.driver.host", "127.0.0.1")
             .config("spark.driver.bindAddress", "127.0.0.1")
             .config("spark.pyspark.python", sys.executable)
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_ratings_u(path=DEFAULT_DATA / "u.data", spark=None):
    from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, LongType
    spark = spark or get_spark_session()
    schema = StructType([
        StructField("userId", IntegerType(), False),
        StructField("movieId", IntegerType(), False),
        StructField("rating", DoubleType(), False),
        StructField("timestamp", LongType(), False),
    ])
    read_ratings(path)  # Validate before Spark reads the file directly.
    return spark.read.option("sep", "\t").option("mode", "FAILFAST").schema(schema).csv(str(path))


def load_movies_u(path=DEFAULT_DATA / "u.item", spark=None, genre_path=None):
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType
    path = Path(path)
    genre_path = genre_path or path.with_name("u.genre")
    spark = spark or get_spark_session()
    read_movies(path, genre_path)
    genres = read_genres(genre_path)
    schema = StructType([
        StructField("movieId", IntegerType(), False),
        StructField("title", StringType(), False),
        StructField("release_date", StringType()),
        StructField("video_release_date", StringType()),
        StructField("imdb_url", StringType()),
    ] + [StructField(f"genre_{index}", IntegerType()) for index in range(19)])
    base = (spark.read.option("sep", "|").option("quote", "")
            .option("encoding", "ISO-8859-1").option("mode", "FAILFAST")
            .schema(schema).csv(str(path)))
    labels = F.array(*[F.when(F.col(f"genre_{index}") == 1, F.lit(genres[index]))
                       for index in range(19)])
    return base.select("movieId", "title", F.filter(labels, lambda label: label.isNotNull()).alias("genres"))


def load_users_u(path=DEFAULT_DATA / "u.user", spark=None):
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType
    spark = spark or get_spark_session()
    schema = StructType([
        StructField("userId", IntegerType()), StructField("age", IntegerType()),
        StructField("gender", StringType()), StructField("occupation", StringType()),
        StructField("zip", StringType()),
    ])
    return spark.read.option("sep", "|").schema(schema).csv(str(path))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
