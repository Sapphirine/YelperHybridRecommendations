from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseNotFound
import pandas_gbq
import pandas
from google.oauth2 import service_account
# import pyspark
# from pyspark.sql import SparkSession
import time
# import pandas as pd
# import geopy
# from geopy.distance import geodesic
# import geopandas as gpd
# import shapely
import numpy as np
import random
# import matplotlib
# import matplotlib.pyplot as plt
from restmap.models import YelpUser

# Create your views here.

# For your purposes, please find the service account file and load it to whichever directory you choose.
credentials = service_account.Credentials.from_service_account_file(r'C:\Users\Owner\Documents\django_app\bigdata-hw01-c97798e29011.json')
#C:\Users\Owner\Documents\django_app\bigdata-hw01-c97798e29011.json

# Initial page, to start with.  Not needed after running initially.  Main page, firstMap will be reloaded after each search
def HomePage(request):
	return render(request, 'homepage.html')

# Rendering of Main Page with Interactive Map and Top 10 Results
def firstMap(request):
	
	pandas_gbq.context.credentials = credentials
	pandas_gbq.context.project = "bigdata-hw01" # Project we pull from
	default_user = 'U5YQX_vMl_xQy8EQDqlNQQ' # Used for generic data, i.e. when no user_id is provided.  Used to get top-rated locations by Yelp star rating.
	
	# What goes from the firstMap HTML template page's input form to Django backend
	realuser = request.GET['query']
	cat = request.GET['category']
	lat = request.GET['lat']
	longit = request.GET['long']
	
	# Booleans used to construct SQL queries
	realuserbool = False
	catbool = False
	latlongbool = False
	
	# Make sure the users you enter are in the project (in the models.py file, should be 5 users to test.
	users = YelpUser.objects.filter(user_id__contains=realuser)
	
	# print("Distinct:", YelpUser.objects.distinct())
	#users = users.filter(user_id__contains=realuser)
	# print(realuser, "type ", type(users))
	# print("User count: ", users.count())
	
	# Initial SQL query statement.  Read through rest of code, it is explanatory in terms of constructing the SQL queries
	FILTER_CONDITION = ""
	filter_count = 0
	
	if (users.count()>=1 and len(realuser)==22 and filter_count == 0): # Pull user recommendations
		realuserbool = True
		FILTER_CONDITION += "WHERE user_id = '{0}' ".format(realuser)
		filter_count += 1
    
	if(cat != ""):
		catbool = True
		cat = cat.lower()
		if(filter_count == 0):
			FILTER_CONDITION = "WHERE categories LIKE '{0}%' ".format(cat)
			filter_count += 1
		else: #(cat != "" and filter_count == 1):
			FILTER_CONDITION += "AND categories LIKE '{0}%' ".format(cat)
			filter_count += 1
      
	latlongbool = lat.replace('.', '', 1).replace('-','',1).isdigit() and longit.replace('.','', 1).replace('-','',1).isdigit()
	if(latlongbool == True):
		maxlat = float(lat) + 2
		minlat = float(lat) - 2
		maxlong = float(longit) + 2
		minlong = float(longit) - 2
		if (maxlat <= 90 and minlat >= -90 and maxlong <= 180 and minlong >= -180):
			latlongbool = True
			if filter_count == 0:
				FILTER_CONDITION += "WHERE latitude > {0} AND latitude < {1} AND longitude > {2} AND longitude < {3}".format(minlat, maxlat, minlong, maxlong)
				filter_count += 1
			else:
				FILTER_CONDITION += "AND latitude > {0} AND latitude < {1} AND longitude > {2} AND longitude < {3}".format(minlat, maxlat, minlong, maxlong)
				filter_count += 1
		else:
			latlongbool = False
	if(realuserbool==True):
		SQL = "SELECT * FROM `bigdata-hw01.Final_Project.recs2` {0} ORDER BY score DESC LIMIT 10".format(FILTER_CONDITION)
	else:
		if(filter_count != 0):
			SQL = "SELECT * FROM `bigdata-hw01.Final_Project.recs2` {0} AND user_id = '{1}' ORDER BY stars DESC LIMIT 10".format(FILTER_CONDITION, default_user)
		else:
			SQL = "SELECT * FROM `bigdata-hw01.Final_Project.recs2` WHERE user_id = '{0}' ORDER BY stars DESC LIMIT 10".format(default_user)
	print("\n----------\n", SQL, "; filter count = ", filter_count, "\n----------\n")
	
	# Constructing the data structures to pass back to the firstMap template.
	data = {}
		
	df = pandas_gbq.read_gbq(SQL)
	df_list = df.to_dict('records')
	data_list = []
	
	if (len(df_list) == 0):
		print("QUERY RETURNED NOTHING!!!!!")
		realuserbool = False
		catbool = False
		latlongbool = False
		SQL = "SELECT * FROM `bigdata-hw01.Final_Project.recs2` WHERE user_id = '{0}' ORDER BY stars DESC LIMIT 10".format(default_user)
		df = pandas_gbq.read_gbq(SQL)
		df_list = df.to_dict('records')
		print("LENGTH OF GENERIC DF: ", len(df_list))
		
	for df_row in df_list:
		temp = {}
		for key in df_row.keys():
			temp[key]=df_row[key]
		data_list.append(temp)
		
	text_entries = []
	text_dict = {'real_user':'','cat':'','lat':'','long':''}
	
	if(realuserbool == True):
		text_dict['real_user'] = realuser
	else:
		text_dict['real_user'] = 'invaliduser'
		
	if(catbool==True):
		text_dict['cat'] = cat
	else:
		text_dict['cat'] = 'empty'
		
	if(latlongbool == True):
		text_dict['lat'] = lat
		text_dict['long'] = float(longit)
	else:
		text_dict['lat'] = 'empty'
		text_dict['long'] = 'empty'
	
	#text_entries.append(text_dict)
	print("\ntext_dict: ", text_dict)
	
	#innerMap = {'data':data_list, 'userid': realuser}
	innerMap = {'data':data_list, 'text_entries':text_dict}
	data = {'response': innerMap}
	
	return render(request, 'firstMap.html', data)

	
	
