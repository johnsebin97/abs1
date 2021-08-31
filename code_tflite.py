# https://drive.google.com/drive/folders/1WrcoRr3bti8lo4A6UrWoV17UrLjxUUDW?usp=sharing

import xgboost as xgb
from keras.applications.vgg16 import VGG16
import numpy as np 
#import matplotlib.pyplot as plt
import cv2

model2 = xgb.XGBClassifier()
model2.load_model("model_xbg1.tflite")


SIZE = 256  #Resize images
#Load model wothout classifier/fully connected layers
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))


#import boto3

#s3 = boto3.resource('s3')
#bucket = s3.Bucket('gradientboostingtrial2')

#fire
#img_path = "WRI/test/fire/Landsat8_20200616_00_4864-3072 - Copy.jpg"
#img_path = "WRI/test/fire/VIIRS_20210813_00_768-768.jpg"
#img_path = "WRI/test/fire/VIIRS_20210813_00_1024-768 - Copy.jpg"
#img_path = "WRI/test/fire/Sent2A20191231_01_0-3584 - Copy.jpg"

# nofire test
#img_path = "WRI/test/nofire/Landsat7_20211216_00_0-0.jpg"
img_path = "3.jpg"

#pimg1 = bucket.Object(img_path).get().get('Body').read()
#pimg1 = img_path
img1 = cv2.imread(img_path, cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, (SIZE, SIZE))
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)


#plt.imshow(img1)

input_img = np.expand_dims(img1, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction = model2.predict(input_img_features)[0] 
#prediction = le.inverse_transform([prediction])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction)
a = "fire" if prediction == 0 else "negative"
print("The prediction for this image is: ", a)

with open('/tmp/gg-amazing-forestfire.log', 'a') as f: print(a, file=f)
