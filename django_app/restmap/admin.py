from django.contrib import admin
from datetime import datetime
import pandas as pd
import pandas_gbq
import time
from google.oauth2 import service_account
import numpy as np
import random
# import matplotlib
# import matplotlib.pyplot as plt

# Register your models here.
# THIS FILE WAS RUN INITIALLY TO LOAD user_ids.  It does not need to be used again for now, unless new users are added.
from restmap.models import YelpUser

class YelpAdmin():
	pass

admin.site.register(YelpUser)

credentials = service_account.Credentials.from_service_account_file(r'C:\Users\Owner\Documents\django_app\bigdata-hw01-c97798e29011.json')
#C:\Users\Owner\Documents\django_app\bigdata-hw01-c97798e29011.json
print("CREDENTIALS: ", credentials)
pandas_gbq.context.credentials = credentials

pandas_gbq.context.project = "bigdata-hw01"

# Get all distinct users from the BigQuery database.
#SQL = "SELECT DISTINCT user_id FROM `homework-1-294021.recommendations.recs`"
SQL = "SELECT DISTINCT user_id FROM `bigdata-hw01.Final_Project.recs2`"


df = pandas_gbq.read_gbq(SQL)
print(df)
df_list = df.to_dict('records')

data_list = []

for df_row in df_list:
	data_list.append({'user_id':df_row['user_id']})
print(len(df_list))
for i in range(len(df_list)):
	print(data_list[i]['user_id'])
	yu = YelpUser(user_id = data_list[i]['user_id']).save()
	
