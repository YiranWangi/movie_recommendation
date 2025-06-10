# utils.py

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

def get_spark_session(app_name="MovieRecommender"):
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

def load_ratings_u(data_dir="data/u.data"):
    schema = StructType([
        StructField("userId", IntegerType()),
        StructField("movieId", IntegerType()),
        StructField("rating", IntegerType()),
        StructField("timestamp", IntegerType()),
    ])
    return get_spark_session().read \
        .option("sep", "\t") \
        .schema(schema) \
        .csv(data_dir) \
        .drop("timestamp")  # 不使用时间戳

def load_movies_u(data_dir="data/u.item"):
    # u.item: movieId | title | release_date | video_release_date | IMDbURL | 19 genre flags
    import pyspark.sql.functions as F
    spark = get_spark_session()
    # 先读基本字段
    base = spark.read.option("sep", "\t") \
        .csv(data_dir) \
        .selectExpr(
            "_c0 as movieId",
            "_c1 as title",
            "_c2 as release_date",
            "_c4 as imdb_url"
        )
    # 读 genre list
    genre_list = spark.read.option("sep", "|").csv("data/u.genre") \
        .filter("_c1 is not null") \
        .selectExpr("_c0 as genre")
    # 读取所有 genre 标志
    flags = spark.read.option("sep", "\t") \
        .csv(data_dir) \
        .select([F.col(f"_c{5+i}").cast("int").alias(g.genre) for i, g in enumerate(genre_list.collect())])
    # 合并 title 和 flags
    movies = base.withColumn("genres", F.array([F.when(flags[c] == 1, c).otherwise(None) for c in flags.columns])) \
                 .select("movieId", "title", "genres")
    return movies

def load_users_u(data_dir="data/u.user"):
    # u.user: userId | age | gender | occupation | zip
    return get_spark_session().read.option("sep", "\t") \
        .csv(data_dir) \
        .selectExpr(
            "_c0 as userId",
            "_c1 as age",
            "_c2 as gender",
            "_c3 as occupation"
        )
