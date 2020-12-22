from pyspark import *
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, lit
from pyspark.ml.recommendation import ALS
import time


def process_review_data(df):
    """ Indexes user, business, and review IDs. """
    cat_cols = ['business_id', 'review_id', 'user_id']
    index_cols = [col+'_ind' for col in cat_cols]
    indexers = [
        StringIndexer(inputCol=col, outputCol=new_col)
        for col, new_col in zip(cat_cols, index_cols)
    ]
    pipeline = Pipeline(stages=indexers)
    model = pipeline.fit(df)
    processed_df = model.transform(df)
    return processed_df


if __name__ == '__main__':

    t = time.time()

    # Spark session
    spark = SparkSession.builder.appName('YelpData').getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set('spark.sql.autoBroadcastJoinThreshold', '-1')

    # Read pre-processed data for businesses
    review_df = spark.read.options(inferSchema=True).json("/home/tanmay/IdeaProjects/BigData_Project/Output/review_df")
    rev_usr_bus = spark.read.options(inferSchema=True).json("/home/tanmay/IdeaProjects/BigData_Project/Output/rev_usr_bus")

    # Index business and user IDs and align with (review) rating.
    user_reviews = review_df.select("review_id", "text").withColumnRenamed("review_id", "r_review_id") \
        .join(rev_usr_bus, col("review_id") == col("r_review_id")) \
        .select("review_id", "business_id", "user_id", "r_stars")
    processed = process_review_data(user_reviews)
    user_reviews_train, user_reviews_test = processed.randomSplit([0.90, 0.1], seed=100)

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(maxIter=5, regParam=0.01, userCol="user_id_ind", itemCol="business_id_ind", ratingCol="r_stars",
              coldStartStrategy="drop")
    model = als.fit(user_reviews_train)

    # Divide the training set to get a sample of how the model works on a sample of the training data.
    sample_train, z = user_reviews_train.randomSplit([0.11, .89], seed=100)
    predictions = model.transform(sample_train)
    final = predictions.withColumn("score", col("prediction") - lit(3)).select("review_id", "business_id", "user_id", "r_stars", "score")
    final.write.json("/home/tanmay/IdeaProjects/BigData_Project/Output/als_similarities_train", mode='overwrite')

    # Get predictions for test data.
    predictions = model.transform(user_reviews_test)
    final = predictions.withColumn("score", col("prediction") - lit(3)).select("review_id", "business_id", "user_id", "r_stars", "score")
    final.write.json("/home/tanmay/IdeaProjects/BigData_Project/Output/als_similarities_test", mode='overwrite')

