
import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

# 设置环境变量（你也可以在 .zshrc 中配置）
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home"
os.environ["SPARK_HOME"] = "/opt/homebrew/Cellar/apache-spark/3.5.0/libexec"

def main():
    # 创建 SparkSession
    spark = SparkSession.builder.appName("电影推荐案例")\
        .master("local[*]")\
        .config("spark.sql.shuffle.partitions", "4")\
        .getOrCreate()

    # 读取数据
    df1 = spark.read.text("data/u.data")
    print("原始数据：")
    df1.show(5, truncate=False)

    # 数据处理函数
    def split_data(line):
        arr = line.split("\t")
        return (int(arr[0]), int(arr[1]), int(arr[2]))

    # 转换为结构化 DataFrame
    df2 = df1.rdd.map(lambda row: split_data(row.value)).toDF(["userId", "movieId", "score"])
    print("处理后的数据：")
    df2.show(5)

    # 划分训练集与测试集
    train_data, test_data = df2.randomSplit([0.8, 0.2])

    # 训练 ALS 模型
    als = ALS(userCol="userId",
              itemCol="movieId",
              ratingCol="score",
              rank=10,
              maxIter=10,
              alpha=1.0,
              coldStartStrategy="drop")  # 避免 NaN 预测

    model = als.fit(train_data)

    # 给某个用户推荐 5 部电影
    df5 = model.recommendForUserSubset(spark.createDataFrame([(653,)], ["userId"]), 5)
    df5.show(truncate=False)

    # 提取推荐结果中的 movieId
    def getMovieIds(row):
        tuijianFilms = []
        for ele in row.recommendations:
            tuijianFilms.append(ele.movieId)
        print("推荐的电影ID有：", tuijianFilms)

    df5.foreach(getMovieIds)

    # 停止 Spark
    spark.stop()

if __name__ == '__main__':
    main()
