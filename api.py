from flask import Flask, request, jsonify
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import explode
from utils import get_spark_session, load_movies

app = Flask(__name__)
# 全局 SparkSession 和模型实例
spark = get_spark_session("APIService")
model = ALSModel.load("models/als_model")
movies = load_movies(spark)

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = int(request.args.get("user_id", 1))
    top_n = int(request.args.get("top_n", 10))

    df_user = spark.createDataFrame([(user_id,)], ["userId"])
    recs = model.recommendForUserSubset(df_user, top_n)

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

    results = recs_with_titles.collect()
    return jsonify([
        {"movieId": r.movieId, "title": r.title, "rating": float(r.rating)}
        for r in results
    ])

if __name__ == "__main__":
    # 启动 Flask 服务
    app.run(host="0.0.0.0", port=5000)
