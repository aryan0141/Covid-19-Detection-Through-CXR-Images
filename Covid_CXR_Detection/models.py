from django.db import models

# Create your models here.
class Images(models.Model): 
    UploadImage = models.ImageField(upload_to='images/') 