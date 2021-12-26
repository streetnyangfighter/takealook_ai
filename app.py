# -*- coding: utf-8 -*-

import img_preprocessing
import img_scoring
import s3_upload
import ai_func
from ast import literal_eval
from flask import Flask, request

import cv2
import json
import collections
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)


@app.route('/cat/face-identify', methods=['POST'])
def test():
    image = img_preprocessing.url_to_img(request.values["orgImg"])

    left_ear_x, left_ear_y, right_ear_x, right_ear_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y, filepath = ai_func.catFaceRecog(
        image)

    left_ear_x = float(left_ear_x)
    left_ear_y = float(left_ear_y)
    right_ear_x = float(right_ear_x)
    right_ear_y = float(right_ear_y)
    left_eye_x = float(left_eye_x)
    left_eye_y = float(left_eye_y)
    right_eye_x = float(right_eye_x)
    right_eye_y = float(right_eye_y)

    url = s3_upload.get_photo_point(filepath)

    data = {'url': url,
            'leftEarX': left_ear_x,
            'leftEarY': left_ear_y,
            'rightEarX': right_ear_x,
            'rightEarY': right_ear_y,
            'leftEyeX': left_eye_x,
            'leftEyeY': left_eye_y,
            'rightEyeX': right_eye_x,
            'rightEyeY': right_eye_y
            }

    return json.dumps(data)


@app.route('/cat/similarity-scoring', methods=['POST'])
def print_string():

    # 새로 등록하는 고양이 이미지 크롭
    url = request.values["url"][1:-1]
    image = img_preprocessing.url_to_img(url)

    left_ear_x = float(request.values["leftEarX"])
    left_ear_y = float(request.values["leftEarY"])
    left_eye_x = float(request.values["leftEyeX"])
    left_eye_y = float(request.values["leftEyeY"])
    right_ear_x = float(request.values["rightEarX"])
    right_ear_y = float(request.values["rightEarY"])
    right_eye_x = float(request.values["rightEyeX"])
    right_eye_y = float(request.values["rightEyeY"])

    croppedImg = img_preprocessing.catFaceCrop(image,
                                               left_ear_x, left_ear_y,
                                               left_eye_x, left_eye_y,
                                               right_ear_x, right_ear_y,
                                               right_eye_x, right_eye_y)

    # 새로 등록하는 고양이 이미지 전처리
    binaryImg1 = img_preprocessing.imgPreprocessing(croppedImg)

    eval = literal_eval(request.values["catList"])

    score_dict = {}
    
    for i in range(len(eval)):

        s3_url = eval[i]["image"]
        image = img_preprocessing.url_to_img(s3_url)

        croppedImg = img_preprocessing.catFaceCrop(image,
                                                   eval[i]["leftEarX"], eval[i]["leftEarY"],
                                                   eval[i]["leftEyeX"], eval[i]["leftEyeY"],
                                                   eval[i]["rightEarX"], eval[i]["rightEarY"],
                                                   eval[i]["leftEyeX"], eval[i]["leftEyeY"])
        
        binaryImg2 = img_preprocessing.imgPreprocessing(croppedImg)
        
        # 스코어 산출
        score = img_scoring.scoring(binaryImg1, binaryImg2)
        
        score_dict[eval[i]["id"]] = score
        
    # 스코어를 기준으로 내림차순 정렬한 key 리스트
    sorted_value = sorted(score_dict.items(), key = lambda x : x[1], reverse=True)
    sorted_dict = collections.OrderedDict(sorted_value) 
    
    data = {'sortedDict': dict(sorted_dict)}

    return json.dumps(data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
