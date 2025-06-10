
# train.py
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from utils import get_spark_session, load_ratings_u

def main():
    spark = get_spark_session("TrainWith_u")
    ratings = load_ratings_u("data/u.data")
    train_data, test_data = ratings.randomSplit([0.8, 0.2], seed=42)

    als = ALS(
        userCol="userId", itemCol="movieId", ratingCol="rating",
        implicitPrefs=False, coldStartStrategy="drop"
    )

    param_grid = ParamGridBuilder() \
        .addGrid(als.rank, [10, 20]) \
        .addGrid(als.maxIter, [10, 20]) \
        .addGrid(als.regParam, [0.01, 0.1]) \
        .build()

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating")
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid,
                        evaluator=evaluator, numFolds=3)
    cv_model = cv.fit(train_data)
    print("Test RMSE:", evaluator.evaluate(cv_model.bestModel.transform(test_data)))

    cv_model.bestModel.save("models/als_model_u")
    spark.stop()
