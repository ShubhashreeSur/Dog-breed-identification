
import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import math
from datetime import datetime
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
from pathlib import Path

from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Dropout,Conv2D,GlobalAveragePooling2D,Flatten
from keras.applications.inception_resnet_v2 import InceptionResNetV2


st.title(" “Dog Breed Identification” : Determine the breed of a dog in an image ")

#loading the image
global X
input_data=st.file_uploader(label='Enter an image of a dog',type=["png","jpg","jpeg"])

if input_data is not None:
    st.image(Image.open(input_data))


#a checkbox to proceed for predictions
segment=st.selectbox('Predict the breed of the dog',['yes','no'],index=1)



if segment == 'yes':


    try:

        X=image.load_img('train/'+str(Path(input_data.name)),target_size=(224,224,3))

        st.write("predicting...")
        st.write()


        #loading the trained files
        with open('label_encoder','rb') as f:
            le=pickle.load(f)

        model=load_model('best_model_inceptionresnetv2')

        start=time.time()

        #converting the array into image and resizing it
        img=image.img_to_array(X)
        test_img=img.reshape((1,224,224,3))


        #predictions
        prediction = model.predict(test_img/255)

        end=time.time()

        #getting the breed and its probability
        predicted_breed=le.inverse_transform([np.argmax(prediction)])[0]
        prob=round(np.max(prediction)*100,3)


        st.write("Predicted breed: ",predicted_breed, 'with a probability of',prob,'%' )


        st.write("\n\nTime required:",round(end-start,3),"secs")


    except Exception as e:
        st.write()




