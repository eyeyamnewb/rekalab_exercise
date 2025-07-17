from django.db import models
from django.contrib.auth.models import User


# Create your models here. 
class fake_doctor(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=260, default="john doe")
    contact = models.IntegerField(default=3434343434)
    email = models.CharField(blank=True, null=True, max_length=260)

    def __str__(self):
        return self.name
    
class fake_specialty(models.Model):
    user = models.ForeignKey(fake_doctor, on_delete=models.CASCADE)
    title = models.CharField( max_length= 40    ,blank=True, null=True)