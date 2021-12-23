import enum
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import * 
import os
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')

# 프론트에서 이미지 파일을 받아서 로컬에 저장을 한 뒤
# 저장된 path 를 받아서 아래 함수 구동
# 고양이 사진에서 얼굴 검출 후, 점 찍어서 result.png 로 저장

# 모델 불러오기
from keras.models import load_model
global model
model = load_model('cat-detection.h5')

def catFaceRecog(image):
    global left_ear_x, left_ear_y, right_ear_x, right_ear_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y
    
    class CatFeatures(enum.Enum):
        # below are the cat facial features (⊙o⊙)
        # eyes
        LEFT_EYE = 0
        RIGHT_EYE = 1
        # mouth
        MOUTH = 2
        # left ear
        LEFT_EAR_1 = 3
        LEFT_EAR_2 = 4
        LEFT_EAR_3 = 5
        # right ear
        RIGHT_EAR_1 = 6
        RIGHT_EAR_2 = 7
        RIGHT_EAR_3 = 8

    def map_labels(labels):
        x = labels[0:18:2]
        y = labels[1:18:2]

        features ={
            CatFeatures.LEFT_EYE : (),
            CatFeatures.RIGHT_EYE : (),
            CatFeatures.MOUTH : (),
            CatFeatures.LEFT_EAR_1 : (),
            CatFeatures.LEFT_EAR_2 : (),
            CatFeatures.LEFT_EAR_3 : (),
            CatFeatures.RIGHT_EAR_1 : (),
            CatFeatures.RIGHT_EAR_2 : (),
            CatFeatures.RIGHT_EAR_3 : (),
                  }
        for key,xpoint,ypoint in zip(features.keys(),x,y):
            features[key] = (xpoint,ypoint)

        return features

    def preprocess_image(image):
        x = image / 255.0
        x = cv2.resize(x,(224,224))
        x = np.asarray(x).astype('float32')
        return x

    def decode_labels(labels,width,heigth):
        labels[0:18:2] = labels[0:18:2] * width
        labels[1:18:2] = labels[1:18:2] * heigth
        return labels

    def predict_image(image, model):
        img = image
        (w, h) = img.shape[:2]

        # predictions
        image_preprocessed = preprocess_image(img)
        y = model.predict(np.expand_dims(image_preprocessed, axis=0)).flatten()
        y = decode_labels(y, w, h)
        return show_cat(img,y)

    def show_cat(image,labels):
        features = map_labels(labels)

        plt.imshow(image)
        plt.axis('off')
        #plt.scatter(labels[0:18:2],labels[1:18:2],c='r',)


        x,y = [],[]
        points = [CatFeatures.LEFT_EAR_1,
                  CatFeatures.LEFT_EAR_2,
                  CatFeatures.LEFT_EAR_3,
                  CatFeatures.RIGHT_EAR_1,
                  CatFeatures.RIGHT_EAR_2,
                  CatFeatures.RIGHT_EAR_3,
                  CatFeatures.MOUTH,
                  CatFeatures.LEFT_EAR_1,
                 ]

        for p in points:
            x.append(features[p][0])
            y.append(features[p][1])

        lines = plt.plot(x,y,marker='*')
        plt.setp(lines, color='c',)

        # 점 찍은 상태의 사진 저장 -> 프론트에 전송
        filepath = "static/result.jpg"
        plt.savefig(filepath)
        # plt.show()

        # 왼쪽 귀 좌표
        left_ear_x = features[CatFeatures.LEFT_EAR_1][0]
        left_ear_y = features[CatFeatures.LEFT_EAR_1][1]

        # 오른쪽 귀 좌표
        right_ear_x = features[CatFeatures.RIGHT_EAR_1][0]
        right_ear_y = features[CatFeatures.RIGHT_EAR_1][1]

        # 왼쪽 눈 좌표
        left_eye_x = features[CatFeatures.LEFT_EYE][0]
        left_eye_y = features[CatFeatures.LEFT_EYE][1]

        # 오른쪽 눈 좌표
        right_eye_x = features[CatFeatures.RIGHT_EYE][0]
        right_eye_y = features[CatFeatures.RIGHT_EYE][1]
        
        return left_ear_x, left_ear_y, right_ear_x, right_ear_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y
    
    return predict_image(image, model)