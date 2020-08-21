# forms.py 
from django import forms 
from .models import *

class Image(forms.ModelForm): 

	class Meta: 
		model = Images 
		#fields = ['name', 'hotel_Main_Img'] 
		fields = ['UploadImage'] 

