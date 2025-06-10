from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import explode
from utils import get_spark_session, load_movies

def recommend_for_user(user_id, top_n=10):
    spark = get_spark_session("Inference")
    # 加载训练好的 ALS 模型
    model = ALSModel.load("models/als_model")
    # 加载电影元数据
    movies = load_movies(spark)

    # 构造单用户 DataFrame
    df_user = spark.createDataFrame([(user_id,)], ["userId"])
    recs = model.recommendForUserSubset(df_user, top_n)

    # 展开 recommendations 列并按评分降序 join 电影表
    recs_expanded = (
        recs
        .selectExpr("explode(recommendations) as rec")
        .selectExpr("rec.movieId as movieId", "rec.rating as rating")
    )
    recs_with_titles = (
        recs_expanded
        .join(movies, "movieId")
        .orderBy("rating", ascending=False)
    )

    recs_with_titles.show(truncate=False)
    spark.stop()

if __name__ == "__main__":
    import sys
    uid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    recommend_for_user(uid)

