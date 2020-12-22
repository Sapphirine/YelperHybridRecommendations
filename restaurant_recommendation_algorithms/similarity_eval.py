import os
from pyspark import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"


def get_decision(x):
    return 1 if x >= 0 else 0


def get_label(x):
    return 1 if x >= 4 else 0


def get_agg_dec(x, y, z):
    return 1 if (x + y + z) >= 2 else 0


udf_get_decision = udf(get_decision, IntegerType())
udf_get_label = udf(get_label, IntegerType())
udf_get_agg_decision = udf(get_agg_dec, IntegerType())

if __name__ == '__main__':

    # Spark Session
    spark = SparkSession.builder.appName('YelpData').getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set('spark.sql.autoBroadcastJoinThreshold', '-1')

    # Read data from training/test results.
    cat_sim = spark.read.options(inferSchema=True).json("/home/tanmay/IdeaProjects/BigData_Project/Output/cat_similarities_test")
    tdf_sim = spark.read.options(inferSchema=True).json("/home/tanmay/IdeaProjects/BigData_Project/Output/tfidf_similarities_test_hl")
    als_sim = spark.read.options(inferSchema=True).json("/home/tanmay/IdeaProjects/BigData_Project/Output/als_similarities_test")

    # Convert recommendation score to binary (0, 1) - where a positive score means the algorithm recommends business.
    d_cat_sim = cat_sim.withColumn("d_cat", udf_get_decision(col("score")))
    d_tdf_sim = tdf_sim.withColumn("d_tdf", udf_get_decision(col("score")))
    d_als_sim = als_sim.withColumn("d_als", udf_get_decision(col("score")))

    # Aggrgate results from attributes, collaborative, and text models.
    agg = d_cat_sim.alias('c').join(d_tdf_sim.alias('t'), col("c.review_id") == col("t.review_id"))\
                              .join(d_als_sim.alias('a'), col("c.review_id") == col("a.review_id"))\
                              .select("c.review_id", "c.business_id", "c.user_id", "c.r_stars", "c.d_cat", "t.d_tdf", "a.d_als")\
                              .filter(col("c.r_stars") != 3).withColumn("label", udf_get_label(col("r_stars")))\
                              .withColumn("d_agg", udf_get_agg_decision(col("c.d_cat"), col("t.d_tdf"), col("a.d_als")))

    # Calculate metrics for content-based filtering - business attributes model.
    cat_labels = agg.select("d_cat", "label").rdd.map(lambda x: (float(x.d_cat), float(x.label)))
    cat_labels_bin = agg.select("d_cat", "label").rdd.map(lambda x: (float(x.d_cat), x.label))
    cat_bin = BinaryClassificationMetrics(cat_labels_bin)
    cat_metrics = MulticlassMetrics(cat_labels)
    cat_precision_1 = cat_metrics.precision(1.0)
    cat_recall_1 = cat_metrics.recall(1.0)
    cat_precision_0 = cat_metrics.precision(0.0)
    cat_recall_0 = cat_metrics.recall(0.0)
    cat_accuracy = cat_metrics.accuracy

    # Calculate metrics for content-based filtering - reviews (text-mining) model.
    tdf_labels = agg.select("d_tdf", "label").rdd.map(lambda x: (float(x.d_tdf), float(x.label)))
    tdf_labels_bin = agg.select("d_tdf", "label").rdd.map(lambda x: (float(x.d_tdf), x.label))
    tdf_bin = BinaryClassificationMetrics(tdf_labels_bin)
    tdf_metrics = MulticlassMetrics(tdf_labels)
    tdf_precision_1 = tdf_metrics.precision(1.0)
    tdf_recall_1 = tdf_metrics.recall(1.0)
    tdf_precision_0 = tdf_metrics.precision(0.0)
    tdf_recall_0 = tdf_metrics.recall(0.0)
    tdf_accuracy = tdf_metrics.accuracy

    # Calculate metrics for collaborative-based filtering - ALS model.
    als_labels = agg.select("d_als", "label").rdd.map(lambda x: (float(x.d_als), float(x.label)))
    als_labels_bin = agg.select("d_als", "label").rdd.map(lambda x: (float(x.d_als), x.label))
    als_bin = BinaryClassificationMetrics(als_labels_bin)
    als_metrics = MulticlassMetrics(als_labels)
    als_precision_1 = als_metrics.precision(1.0)
    als_recall_1 = als_metrics.recall(1.0)
    als_precision_0 = als_metrics.precision(0.0)
    als_recall_0 = als_metrics.recall(0.0)
    als_accuracy = als_metrics.accuracy

    # Aggrgate labels using majority (best-of-3) rule.
    agg_labels = agg.select("d_agg", "label").rdd.map(lambda x: (float(x.d_agg), float(x.label)))
    agg_labels_bin = agg.select("d_agg", "label").rdd.map(lambda x: (float(x.d_agg), x.label))
    agg_bin = BinaryClassificationMetrics(agg_labels_bin)
    agg_metrics = MulticlassMetrics(agg_labels)
    agg_precision_1 = agg_metrics.precision(1.0)
    agg_recall_1 = agg_metrics.recall(1.0)
    agg_precision_0 = agg_metrics.precision(0.0)
    agg_recall_0 = agg_metrics.recall(0.0)
    agg_accuracy = agg_metrics.accuracy

    # Print metrics.
    print("Total Recommendations: ", "36240")
    print("Category Precision 1: ", cat_precision_1)
    print("Category Recall 1: ", cat_recall_1)
    print("Category Accuracy: ", cat_accuracy)
    print("TF-IDF Precision 1: ", tdf_precision_1)
    print("TF-IDF Recall 1: ", tdf_recall_1)
    print("TF-IDF Accuracy: ", tdf_accuracy)
    print("ALS Precision 1: ", als_precision_1)
    print("ALS Recall 1: ", als_recall_1)
    print("ALS Accuracy: ", als_accuracy)
    print("Aggregate Precision 1: ", agg_precision_1)
    print("Aggregate Recall 1: ", agg_recall_1)
    print("Aggregate Accuracy 0: ", agg_accuracy)

