import os
import time
from pyspark import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, collect_list, lit
from pyspark.sql.types import StringType, ArrayType, DoubleType
from pyspark.ml.feature import HashingTF, IDF, Normalizer
import nltk
from nltk.stem import WordNetLemmatizer


os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"


def read_file(datatype):
    """ Read raw Yelp data from .json to Spark DataFrame."""
    start = time.time()
    print("Reading {} file...".format(datatype[0]))
    df = spark.read.json(datatype[1])
    print("Read {}. Time elapsed: {}".format(datatype[0], time.time() - start))
    return df


def get_nouns_from_text(row):
    """ Make words lower case, tokenize, """
    global wordnet_lemmatizer
    reviews = row.lower()
    tokenized = nltk.word_tokenize(reviews)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if (pos[:2] == 'NN')]
    lemmatized_words = [wordnet_lemmatizer.lemmatize(word) for word in nouns]
    return lemmatized_words


def concat_rev_lists(row):
    """ Concatenates a list of text reviews into a single list of words."""
    conc = list()
    [conc.extend(x) for x in row]
    return conc


def dot_product(x, y):
    try:
        return float(x.dot(y))
    except (TypeError, AttributeError):
        return 0


# GLOBALS
# Create Spark SQL user-defined functions for functions defined above.
wordnet_lemmatizer = WordNetLemmatizer()
udf_nouns_from_text = udf(get_nouns_from_text, ArrayType(StringType()))
udf_concat_rev_lists = udf(concat_rev_lists, ArrayType(StringType()))
udf_dot_prod = udf(dot_product, DoubleType())

if __name__ == '__main__':

    # Start time
    t = time.time()
    # Spark Session
    spark = SparkSession.builder.appName('YelpData').getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set('spark.sql.autoBroadcastJoinThreshold', '-1')

    # Load processed data - reviews and user-business-review dataframes
    review_df = spark.read.options(inferSchema=True).json("/home/tanmay/IdeaProjects/BigData_Project/Output/review_df")
    rev_usr_bus = spark.read.options(inferSchema=True).json("/home/tanmay/IdeaProjects/BigData_Project/Output/rev_usr_bus")
    user_reviews = review_df.select("review_id", "text").withColumnRenamed("review_id", "r_review_id") \
        .join(rev_usr_bus, col("review_id") == col("r_review_id")) \
        .select("review_id", "business_id", "user_id", "r_stars", "text")
    # Break into training and test set.
    user_reviews_train, user_reviews_test = user_reviews.randomSplit([0.90, 0.1], seed=100)

    # ---------------------------------------------------------------------------------------------------------------- #
    # TF-IDF
    # Initalize objects for HashingTF, Inverse Document Frequency, and Vector Normalizer.
    hashing_tf = HashingTF(inputCol="nouns", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    normalizer = Normalizer(inputCol="features", outputCol="normFeatures")

    # Process train and test reviews text to find nouns (lemmatized).
    nouns_df = user_reviews.withColumn("nouns", udf_nouns_from_text(col("text"))).drop("text")
    nouns_df_train = user_reviews_train.withColumn("nouns", udf_nouns_from_text(col("text"))).drop("text")
    # Concatenate text of all good (bad) business reviews by business ID to build business profile.
    bus_rev_df_high = nouns_df.filter(col("r_stars") >= 4).groupby("business_id")\
                              .agg(collect_list(col("nouns")).alias("text")).drop("nouns")\
                              .withColumn("nouns", udf_concat_rev_lists(col("text"))).drop("text")
    bus_rev_df_low = nouns_df.filter(col("r_stars") <= 2).groupby("business_id")\
                             .agg(collect_list(col("nouns")).alias("text")).drop("nouns")\
                             .withColumn("nouns", udf_concat_rev_lists(col("text"))).drop("text")
    # Concatenate text of all good (bad) user reviews by user ID to build user profile.
    user_rev_high = nouns_df_train.filter(col("r_stars") >= 4).groupby("user_id")\
                                  .agg(collect_list(col("nouns")).alias("text")).drop("nouns")\
                                  .withColumn("nouns", udf_concat_rev_lists(col("text"))).drop("text")
    user_rev_low = nouns_df_train.filter(col("r_stars") <= 2).groupby("user_id")\
                                 .agg(collect_list(col("nouns")).alias("text")).drop("nouns")\
                                 .withColumn("nouns", udf_concat_rev_lists(col("text"))).drop("text")
    # Create final business and user dataframes
    cols = ['business_id', 'user_id', 'nouns']
    temp1_1 = bus_rev_df_high.withColumn("user_id", lit(None)).select(cols)
    temp1_2 = bus_rev_df_low.withColumn("user_id", lit(None)).select(cols)
    temp2 = user_rev_high.withColumn("business_id", lit(None)).select(cols)
    temp3 = user_rev_low.withColumn("business_id", lit(None)).select(cols)

    # Union of business and user data for good reviews.
    rev_high = temp1_1.union(temp2)
    # Hash each word in vocabulary and find frequency of word.
    hashed_high = hashing_tf.transform(rev_high).drop("nouns")
    # Fit TF-IDF model to the training data.
    idf_high = idf.fit(hashed_high)
    rescaled_high = idf_high.transform(hashed_high).drop("rawFeatures")
    # Normalize each (row) TF-IDF feature vector.
    norm_high = normalizer.transform(rescaled_high)

    # Union of business and user data for bad reviews.
    rev_low = temp1_2.union(temp3)
    # Hash each word in vocabulary and find frequency of word.
    hashed_low = hashing_tf.transform(rev_low).drop("nouns")
    # Fit TF-IDF model to the training data.
    idf_low = idf.fit(hashed_low)
    rescaled_low = idf_low.transform(hashed_low).drop("rawFeatures")
    # Normalize each (row) TF-IDF feature vector.
    norm_low = normalizer.transform(rescaled_low)

    # Rename columns.
    bh = norm_high.withColumnRenamed("normFeatures", "bh_normFeatures")
    uh = norm_high.withColumnRenamed("normFeatures", "uh_normFeatures")
    bl = norm_low.withColumnRenamed("normFeatures", "bl_normFeatures")
    ul = norm_low.withColumnRenamed("normFeatures", "ul_normFeatures")

    # Divide the training set to get a sample of how the model works on training data.
    sample_train, z = user_reviews_train.randomSplit([0.11, 0.89], seed=100)
    rev_score = sample_train.alias('r').join(bh.alias('bh'), col("r.business_id") == col("bh.business_id"), how='left') \
        .join(uh.alias('uh'), col("r.user_id") == col("uh.user_id"), how='left') \
        .join(bl.alias('bl'), col("r.business_id") == col("bl.business_id"), how='left') \
        .join(ul.alias('ul'), col("r.user_id") == col("ul.user_id"), how='left') \
        .select("r.review_id", "r_stars", "r.business_id", "r.user_id",
                "bh_normFeatures", "uh_normFeatures", "bl_normFeatures", "ul_normFeatures")
    # Calculate similarity score between a business and user in the traning set.
    similarities = rev_score.withColumn("high_similarity", udf_dot_prod(col("bh_normFeatures"), col("uh_normFeatures"))) \
        .withColumn("low_similarity", udf_dot_prod(col("bl_normFeatures"), col("ul_normFeatures"))) \
        .fillna({"high_similarity": 0, "low_similarity": 0}) \
        .withColumn("score", col("high_similarity") - col("low_similarity"))
    final = similarities.select("review_id", "business_id", "user_id", "r_stars", "score")
    final.write.json("/home/tanmay/IdeaProjects/BigData_Project/Output/tfidf_similarities_train", mode='overwrite')
    del rev_score, similarities, final

    # Recommend a business (in the test set) to a user on the basis of user-business similarity score as per user
    # and business review (documents) topics of the training set.
    rev_score = user_reviews_test.alias('r').join(bh.alias('bh'), col("r.business_id") == col("bh.business_id"), how='left')\
                                       .join(uh.alias('uh'), col("r.user_id") == col("uh.user_id"), how='left') \
                                       .join(bl.alias('bl'), col("r.business_id") == col("bl.business_id"), how='left')\
                                       .join(ul.alias('ul'), col("r.user_id") == col("ul.user_id"), how='left') \
                                       .select("r.review_id", "r_stars", "r.business_id", "r.user_id",
                                               "bh_normFeatures", "uh_normFeatures", "bl_normFeatures", "ul_normFeatures")
    # Calculate similarity score between user-business (using training data) for each business a user has
    # visited in the test set (user reviews in the test set are not used to create the user's profile).
    similarities = rev_score.withColumn("high_similarity", udf_dot_prod(col("bh_normFeatures"), col("uh_normFeatures")))\
                            .withColumn("low_similarity", udf_dot_prod(col("bl_normFeatures"), col("ul_normFeatures")))\
                            .fillna({"high_similarity": 0, "low_similarity": 0})\
                            .withColumn("score", col("high_similarity") - col("low_similarity"))
    final = similarities.select("review_id", "business_id", "user_id", "r_stars", "score")
    final.write.json("/home/tanmay/IdeaProjects/BigData_Project/Output/tfidf_similarities_test", mode='overwrite')

    # Record and print execution time.
    t = time.time() - t
    print(t)
