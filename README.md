# Yelper: Hybrid Recommendation System
EECS 6893: Big Data Analytics Final Project

Contributors: Tanmay Shah (tanmay.shah@columbia.edu), Riddhima Narravula (rrn2119@columbia.edu), Deepak Dwarakanath (dd2676@columbia.edu)

Dataset: https://www.yelp.com/dataset

Our project goal was to build a hybrid recommendation system to recommend restaurants to users based on their Yelp reviews. In addition to finding restaurants that users may like, we also aimed to identify restaurants that users may dislike. We used a combination of collaborative filtering and content-based filtering to create recommendations. Specifically, our content-based filtering approaches used categorical data about businesses and textual data from reviews to help recommend restaurants based on what users actually like about a business. After aggregating the results from our individual algorithms, we were able to create a hybrid recommendation system and display our results using a Django-based web application. 

Above you can find our source code: 
- restaurant_recommendation_algorithms: contains code for each of our filtering algorithms (ALS collaborative filtering, content-based filtering with categories, and content-based filtering with review text) as well as code that evaluates and processes our data
- django_app: contains our Django webapp code that produces a visualization of the results from our recommendation algorithm
- attempted_collab_filtering_methods: attempts at collaborative filtering using other methods (not evaluated or used for our final results)
