import os
import time
from pyspark import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array
from pyspark.sql.types import DoubleType
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine
from RestaurantRecommender.process_business_data import process_bus_data, categories

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"


def read_file(datatype):
    """ Read raw Yelp data from .json to Spark DataFrame."""

    start = time.time()
    print("Reading {} file...".format(datatype[0]))
    df = spark.read.json(datatype[1])
    print("Read {}. Time elapsed: {}".format(datatype[0], time.time() - start))
    return df


def similarity(x, y):
    """ Compute cosine similarity score between user likes/dislikes and business attributes"""
    try:
        # Cosine function here gives cosine distance, so we convert to similarity.
        return -float(cosine(x, y)) + 1
    except (TypeError, AttributeError):
        return 0


# GLOBALS
bucket = "/home/tanmay/IdeaProjects/BigData_Project/yelp_dataset"    # Replace with own bucket name
business_json_path = '{}/yelp_academic_dataset_business.json'.format(bucket)
user_json_path = '{}/yelp_academic_dataset_user.json'.format(bucket)
review_json_path = '{}/yelp_academic_dataset_review.json'.format(bucket)
wordnet_lemmatizer = WordNetLemmatizer()
# Define similarity() as a Spark SQL user-defined function.
udf_similarity = udf(similarity, DoubleType())


if __name__ == '__main__':

    spark = SparkSession.builder.appName('YelpData').getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set('spark.sql.autoBroadcastJoinThreshold', '-1')

    # Set coordinate range to filter businesses by location.
    lat_min = 40
    lat_max = 45
    lon_min = -80
    lon_max = -73

    # ---------------------------------------------------------------------------------------------------------------- #
    # Data Processing

    # Read raw data.
    business = ['Business Data', business_json_path]
    bus_df = read_file(business).filter(col("attributes").isNotNull())\
                                .filter(col("latitude") > lat_min).filter(col("latitude") < lat_max)\
                                .filter(col("longitude") > lon_min).filter(col("longitude") < lon_max)
    bus_df.take(1)
    # Record schema of business data.
    bus_schema = bus_df.schema
    # Process business data for content-based features.
    features, n_features = process_bus_data(bus_df)
    # Save features to json.
    features.write.json("/home/tanmay/IdeaProjects/BigData_Project/Output/bus_cat", mode='overwrite')

    # Collect list of business IDs eligible by category and location.
    bus_list = features.select("business_id").rdd.map(lambda x: x.business_id).collect()

    # Read review data into dataframes
    review = ['Review Data', review_json_path]
    temp = read_file(review)
    # Select reviews only for eligible businesses.
    review_df = temp.select("review_id", "user_id", "business_id", "stars", "text")\
                    .filter(col("business_id").isin(bus_list))
    # Save processed review data.
    review_df.write.json("/home/tanmay/IdeaProjects/BigData_Project/Output/review_df", mode='overwrite')

    # List of users with 10 or more reviews.
    user_list = review_df.groupby("user_id").count().filter(col("count") >= 10)

    # Join review, business, and user data.
    rev_bus = review_df.withColumnRenamed("business_id", "r_business_id") \
        .withColumnRenamed("stars", "r_stars") \
        .join(features, col("r_business_id") == col("business_id")) \
        .select(["review_id", "user_id", "business_id", "r_stars"] + ["f[{}]".format(i) for i in range(n_features)])
    rev_usr_bus = user_list.select("user_id")\
        .withColumnRenamed("user_id", "u_user_id")\
        .join(rev_bus, col("u_user_id") == col("user_id")).drop("u_user_id")

    # Save processed review, user, and business data.
    rev_usr_bus.write.json("/home/tanmay/IdeaProjects/BigData_Project/Output/rev_usr_bus", mode='overwrite')

    # ---------------------------------------------------------------------------------------------------------------- #
    # Content-based filtering.

    # Join user reviews with other business features.
    user_reviews = review_df.select("review_id", "text").withColumnRenamed("review_id", "r_review_id") \
        .join(rev_usr_bus, col("review_id") == col("r_review_id")) \
        .select("review_id", "business_id", "user_id", "r_stars", "text")

    # Split data into train and test set.
    user_reviews_train, user_reviews_test = user_reviews.randomSplit([0.90, 0.1], seed=100)

    # Train content based model.
    train_data = user_reviews_train.alias('u').join(rev_usr_bus.alias('r'), [col("u.user_id") == col("r.user_id"),
                                                                  col("u.business_id") == col("r.business_id"),
                                                                  col("u.review_id") == col("r.review_id")])\
                                              .select("u.user_id", *["f[{}]".format(i) for i in range(n_features)])

    # Split user-business preferences into liked and disliked.
    user_high = rev_usr_bus.filter(col("r_stars") >= 4).groupby("user_id").mean(*["f[{}]".format(i) for i in range(n_features)])
    user_low = rev_usr_bus.filter(col("r_stars") <= 2).groupby("user_id").mean(*["f[{}]".format(i) for i in range(n_features)])
    # Create user-feature-vector by averaging across all (liked/disliked) businesses reviewed.
    a = ["avg(f[{}])".format(i) for i in range(n_features)]
    user_high_features = user_high.select("user_id", array(*a).alias("features"))
    user_low_features = user_low.select("user_id", array(*a).alias("features"))
    bus_features = features.select("business_id", "stars",  array(*["f[{}]".format(i) for i in range(n_features)]).alias("features"))
    # Final feature vectors for businesses, user-likes, user-dislikes
    b = bus_features.withColumnRenamed("features", "b_features")
    uh = user_high_features.withColumnRenamed("features", "uh_features")
    ul = user_low_features.withColumnRenamed("features", "ul_features")

    # Divide the training set to get a sample of how the model works on training data.
    sample_train, z = user_reviews_train.randomSplit([0.11, 0.89], seed=100)
    rev_score_train = sample_train.alias('r').join(b.alias('b'), col("r.business_id") == col("b.business_id"), how='left') \
        .join(uh.alias('uh'), col("r.user_id") == col("uh.user_id"), how='left') \
        .join(ul.alias('ul'), col("r.user_id") == col("ul.user_id"), how='left') \
        .select("r.review_id", "r_stars", "r.business_id", "stars", "r.user_id",
                "b_features", "uh_features", "ul_features")
    # Calculate similarity score between a business and user in the traning set.
    similarities = rev_score_train.withColumn("high_similarity", udf_similarity(col("b_features"), col("uh_features"))) \
        .withColumn("low_similarity", udf_similarity(col("b_features"), col("ul_features"))) \
        .fillna({"high_similarity": 0, "low_similarity": 0}) \
        .withColumn("score", col("high_similarity") - col("low_similarity"))
    final = similarities.select("review_id", "business_id", "user_id", "r_stars", "score")
    final.write.json("/home/tanmay/IdeaProjects/BigData_Project/Output/cat_similarities_train_pca13", mode='overwrite')
    del similarities, final, rev_score_train

    # Recommend a business (in the test set) to a user on the basis of user-business similarity score as per user
    # and business profile of the training set.
    rev_score_test = user_reviews_test.alias('r').join(b.alias('b'), col("r.business_id") == col("b.business_id"), how='left') \
        .join(uh.alias('uh'), col("r.user_id") == col("uh.user_id"), how='left') \
        .join(ul.alias('ul'), col("r.user_id") == col("ul.user_id"), how='left') \
        .select("r.review_id", "r_stars", "r.business_id", "stars", "r.user_id",
                "b_features", "uh_features", "ul_features")
    # Calculate similarity score between user-business (using training data) for each business a user has
    # visited in the test set (user reviews in the test set are not used to create the user's profile).
    similarities = rev_score_test.withColumn("high_similarity", udf_similarity(col("b_features"), col("uh_features"))) \
        .withColumn("low_similarity", udf_similarity(col("b_features"), col("ul_features"))) \
        .fillna({"high_similarity": 0, "low_similarity": 0}) \
        .withColumn("score", col("high_similarity") - col("low_similarity"))
    final = similarities.select("review_id", "business_id", "user_id", "r_stars", "score")
    final.write.json("/home/tanmay/IdeaProjects/BigData_Project/Output/cat_similarities_test_pca13", mode='overwrite')


