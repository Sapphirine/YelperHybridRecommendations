from django.db import models
import uuid

# Create your models here.

class YelpUser(models.Model):
	user_id = models.CharField(max_length=22, unique=True)
	
def __str__(self):
	return self.user_id
