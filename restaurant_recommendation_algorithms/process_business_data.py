import numpy as np
import pandas as pd
from pyspark import *
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, IntegerType, ArrayType, BooleanType
from pyspark.ml.feature import CountVectorizer, VectorAssembler, PCA

from scipy.sparse import csc_matrix
from pyspark.ml.linalg import _convert_to_vector, VectorUDT
from pyspark.ml.functions import vector_to_array

categories = pd.read_csv("/home/tanmay/IdeaProjects/BigData_Project/categories.csv", header=None)[0].tolist()
categories.sort()


def process_bus_data(bus_df):
    """ Method to process raw business data from Yelp."""

    def select_elibigble_bus(row):
        """ Select businesses which fall into selected categores."""

        global categories
        try:
            # Return true if business falls into category list, else false.
            row_cats = row.split(',')
            for cat in row_cats:
                if cat.strip() in categories:
                    return True
            return False
        except (TypeError, AttributeError):
            # Returns false if business has no defined categories.
            return False

    def unpack_bus_attributes(row):
        """ Unpacks Business attributes and assigns them an index value."""

        # List to store business attributes.
        unpacked = list()
        # Unpack all attributes except PriceRange and Parking
        temp = [row[s] for s in bus_attributes]

        # Process PriceRange
        try:
            priceRange = int(row["RestaurantsPriceRange2"])
        except (TypeError, ValueError):
            # If no price range specified - default=2
            priceRange = 2

        #Process Parking
        try:
            parking = 1 if (row["BusinessParking"].find("True")) != -1 else -1
        except AttributeError:
            parking = 0

        # Process WiFi
        if row["WiFi"] == 'no' or row["WiFi"] == "u'no'":
            wifi = -1
        elif row["WiFi"] == None:
            wifi = 0
        else:
            wifi = 1

        # Tokenize all Boolean attributes.
        for i in temp:
            if i == "True":
                unpacked.append(1)
            elif i == "False":
                unpacked.append(-1)
            else:
                unpacked.append(0)
        # Append the Parking and PriceRange attributes
        unpacked.append(wifi)
        unpacked.append(parking)
        unpacked.append(priceRange)

        # Print any arrays that are not of desired length (=30).
        if len(unpacked) != 30:
            print(unpacked)
        return _convert_to_vector(csc_matrix(np.asarray(unpacked).astype(float)).T)

    def unpack_bus_categories(row):
        """Unpacks all business cattegories."""

        # List to store business categories.
        unpacked = list()
        # Unpack all attributes except PriceRange and Parking
        for cat in row.split(','):
            unpacked.append(cat.strip())
        return unpacked

    def unpack_price_range(row):
        """ Returns price range."""
        return int(row[-1])

    # Package the functions above into Spark SQL user-defined functions
    udf_select_eligible_bus = udf(select_elibigble_bus, BooleanType())
    udf_unpack_bus_attributes = udf(unpack_bus_attributes, VectorUDT())
    udf_unpack_bus_categories = udf(unpack_bus_categories, ArrayType(StringType()))
    udf_unpack_price_range = udf(unpack_price_range, IntegerType())

    # Find businesses to include.
    eligible_bus = bus_df.withColumn("include", udf_select_eligible_bus(col("categories"))) \
        .filter(col("include") == True)

    # Process business attributes feature.
    all_bus_attributes = set(bus_df.select("attributes").take(1)[0].attributes.asDict().keys())
    bus_attributes_to_exclude = {'AcceptsInsurance', 'AgesAllowed', 'ByAppointmentOnly', 'Caters', 'Corkage',
                                 'DietaryRestrictions', 'HairSpecializesIn', 'Open24Hours', 'RestaurantsAttire',
                                 'RestaurantsPriceRange2', 'BusinessParking', 'WiFi'}
    bus_attributes = list(all_bus_attributes - bus_attributes_to_exclude)
    bus_attributes.sort()
    eligible_attr = eligible_bus.withColumn("unpackedAttr", udf_unpack_bus_attributes(col("attributes")))

    # Process business categories feature.
    eligible_cats = eligible_attr.withColumn("unpackedCats", udf_unpack_bus_categories(col("categories")))
    cv = CountVectorizer(inputCol="unpackedCats", outputCol="vectorizedCats")
    vectorized_cats = cv.fit(eligible_cats).transform(eligible_cats)

    # Un-bundle price range from all other attributes.
    unpacked_pr = vectorized_cats.withColumn("priceRange", udf_unpack_price_range(col("unpackedAttr")))
    unpacked_pr.take(1)

    # Reduce dimensions of attributes and categories features, respectively.
    pca_attr = PCA(k=3, inputCol="unpackedAttr", outputCol="pcaAttr").fit(unpacked_pr)
    temp = pca_attr.transform(unpacked_pr)
    temp.show()
    pca_cats = PCA(k=1, inputCol="vectorizedCats", outputCol="pcaCats").fit(temp)
    temp2 = pca_cats.transform(temp)
    temp2.show()

    # Assemble into final feature vector.
    va = VectorAssembler(inputCols=["stars", "priceRange", "pcaAttr", "pcaCats"], outputCol="featureVec")
    features = va.transform(temp2).select("business_id", "stars", "categories", "featureVec")
    features.take(1)

    # Unpack
    n_features = len(features.select("featureVec").take(1)[0].featureVec)
    final = features.withColumn("f", vector_to_array(col("featureVec"))) \
        .select(["business_id", "stars", "categories"] + [col("f")[i] for i in range(n_features)])

    return final, n_features

