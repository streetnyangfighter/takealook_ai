import cv2
import numpy as np
import requests

# s3 에 업로드 된 이미지 읽어오기
def url_to_img(url):
    image_nparray = np.asarray(
    bytearray(requests.get(url).content), dtype=np.uint8)
    image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    
    return image

# 고양이 이마 크롭
def catFaceCrop(image,
                left_ear_x, left_ear_y, 
                left_eye_x, left_eye_y,
                right_ear_x, right_ear_y, 
                right_eye_x, right_eye_y):
    
    img = image

    # [x,y] 좌표점을 4x2의 행렬로 작성
    # 좌표점은 좌상->좌하->우상->우하
    pts1 = np.float32([[left_ear_x,left_ear_y],[left_eye_x,left_eye_y],[right_ear_x,right_ear_y],[right_eye_x,right_eye_y]])

    # 좌표의 이동점
    pts2 = np.float32([[10,10],[10,1000],[1000,10],[1000,1000]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (1100,1100))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    
    return dst


# 고양이 이마 사진 전처리
    # 1. 이미지 흑백화
    # 2. 이미지 평활화
    # 3. 이미지 이진화
def imgPreprocessing(croppedImg):    
    origin_image = croppedImg

    gray_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

    eqHist_image = cv2.equalizeHist(gray_image)

    border01, binaryImg = cv2.threshold(eqHist_image, 150, 255, cv2.THRESH_BINARY)
    
    return binaryImg