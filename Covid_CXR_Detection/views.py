from django.http import HttpResponse
from django.shortcuts import render, redirect 
from .models import Images
import cv2
from imutils import rotate_bound
from os import listdir,remove
from os.path import isfile, join
from keras.models import load_model
import joblib
import numpy as np
import pandas as pd

print("[INFO] loading the trained networks...")
cnn_model = load_model("trained_models/new_model.h5")
rfc_model = joblib.load("trained_models/random_forest_model")

# Create your views here.
def index(request): 
    if request.method == 'POST': 
        # form = Image(request.POST, request.FILES)
        pic = request.FILES['image']
        print(pic)
        image = Images(UploadImage = pic)
        image.save()
        # if form.is_valid(): 
        data_path = 'images'
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
        img=cv2.imread(f"images/{onlyfiles[0]}")
        img=cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
        print(img.shape)
        img = np.array(img).astype('float32')/255
        img = np.expand_dims(img, axis=0)

        label_rfc = cnn_model.predict(img)
        label_rfc = pd.DataFrame(data=label_rfc)
        label_rfc = rfc_model.predict_proba(label_rfc)

        prob_normal = "{:.2f}".format(label_rfc[0][0]*100)
        prob_covid = "{:.2f}".format(label_rfc[0][1]*100)
        prob_pneumonia = "{:.2f}".format(label_rfc[0][2]*100)

        if float(prob_normal)>=80:
            final_output = "No Infection detected"
        elif float(prob_covid)>=80:
            final_output = "Coronavirus Infection detected"
        elif float(prob_pneumonia)>=80:
            final_output = "Pneomonia or Other Bacterial Infection detected"
        else:
            final_output = "Not able to detect the final output"

        params = {'normal':prob_normal, 
                'covid':prob_covid, 
                'pneumonia': prob_pneumonia, 
                'final':final_output
        }
        print(params)
        remove(f"images/{onlyfiles[0]}")	 	

        return render(request,'result.html', params)  
    else: 
        return render(request,'index.html') 
  
  
def success(request): 
    return render(request,'success.html')
