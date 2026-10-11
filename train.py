"""Tune explicit-rating ALS on validation data and evaluate a held-out test split."""

import argparse
import hashlib
import math
import time
from pathlib import Path

from utils import DEFAULT_DATA, DEFAULT_RUN, get_spark_session, load_ratings_u, write_json


def evaluate(model, ratings, train_mean):
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql import functions as F

    total = ratings.count()
    predictions = model.transform(ratings).filter(
        F.col("prediction").isNotNull() & ~F.isnan("prediction")
    ).cache()
    scored = predictions.count()
    if not total or not scored:
        predictions.unpersist()
        raise ValueError("Evaluation split has no scorable ratings; use more data.")
    evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
    metrics = {
        "total_rows": total, "scored_rows": scored, "unscored_rows": total - scored,
        "coverage": scored / total,
        "als_rmse": evaluator.evaluate(predictions),
        "als_mae": evaluator.setMetricName("mae").evaluate(predictions),
        "global_mean_rmse_on_same_rows": evaluator.setMetricName("rmse").evaluate(
            predictions.withColumn("prediction", F.lit(train_mean))),
        "global_mean_rmse_on_all_rows": evaluator.evaluate(
            ratings.withColumn("prediction", F.lit(train_mean))),
    }
    predictions.unpersist()
    return metrics


def train(data_dir, output_dir, ranks, reg_params, max_iter=10, seed=42):
    import pyspark
    from pyspark.ml.recommendation import ALS
    from pyspark.sql import functions as F

    output_dir, data_dir = Path(output_dir), Path(data_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Use a new or empty output_dir.")
    started = time.perf_counter()
    spark = get_spark_session("MovieLensALSTraining")
    try:
        ratings = load_ratings_u(data_dir / "u.data", spark).orderBy("userId", "movieId").cache()
        total = ratings.count()
        training, validation, test = [
            part.cache() for part in ratings.randomSplit([0.8, 0.1, 0.1], seed=seed)
        ]
        split_sizes = {"train": training.count(), "validation": validation.count(), "test": test.count()}
        if min(split_sizes.values()) == 0:
            raise ValueError("Empty split; use more ratings.")
        train_mean = float(training.agg(F.avg("rating")).first()[0])
        best_model, best_rmse = None, math.inf
        grid, best_parameters = [], None
        for rank in ranks:
            for reg_param in reg_params:
                parameters = {"rank": rank, "regParam": reg_param, "maxIter": max_iter}
                model = ALS(
                    **parameters, seed=seed, userCol="userId", itemCol="movieId",
                    ratingCol="rating", implicitPrefs=False, coldStartStrategy="drop",
                    numUserBlocks=2, numItemBlocks=2
                ).fit(training)
                metrics = evaluate(model, validation, train_mean)
                grid.append({"parameters": parameters, "validation": metrics})
                print(f"Validation rank={rank}, reg={reg_param}: "
                      f"RMSE={metrics['als_rmse']:.4f}, coverage={metrics['coverage']:.3f}")
                if metrics["als_rmse"] < best_rmse:
                    best_model, best_rmse, best_parameters = model, metrics["als_rmse"], parameters
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model.save(str(output_dir / "model"))
        test_metrics = evaluate(best_model, test, train_mean)
        report = {
            "algorithm": "Spark explicit-feedback ALS", "pyspark_version": pyspark.__version__,
            "seed": seed, "rating_rows": total, "split_sizes": split_sizes,
            "split_policy": "seeded random rating split: 80/10/10",
            "selection_metric": "validation RMSE on scorable rows",
            "train_mean_rating": train_mean, "validation_grid": grid,
            "best_parameters": best_parameters, "test": test_metrics,
            "data_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                            for name in ("u.data", "u.item", "u.genre")},
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "metrics.json", report)
        print("Held-out test:", test_metrics)
        return report
    finally:
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--ranks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--reg_params", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_iter < 1 or min(args.ranks) < 1 or any(
        not math.isfinite(value) or value <= 0 for value in args.reg_params
    ):
        parser.error("Ranks, max_iter and finite reg_params must be positive.")
    try:
        train(args.data_dir, args.output_dir, args.ranks, args.reg_params, args.max_iter, args.seed)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
