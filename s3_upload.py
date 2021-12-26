# -*- coding: utf-8 -*-

import boto3
from file_manager import AWS_BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY
import os, uuid

def get_photo_point(path):
    s3 = s3_connection();
    
    # filename = "/ai/trial.jpg"
    filename =  "ai/" + str(uuid.uuid4()) + "trial.jpg"
    ret = s3_put_object(s3, AWS_BUCKET_NAME, path, filename)
    if ret : 
        print("파일 저장 성공")
        if os.path.exists(path):
            os.remove(path)
    else :
        print("파일 저장 실패")
    	
    return s3_get_image_url(s3, filename)
    
    

def s3_connection():
    try:
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2", # 자신이 설정한 bucket region
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return s3


def s3_put_object(s3, bucket, filepath, access_key):
    """
    s3 bucket에 지정 파일 업로드
    :param s3: 연결된 s3 객체(boto3 client)
    :param bucket: 버킷명
    :param filepath: 파일 위치
    :param access_key: 저장 파일명
    :return: 성공 시 True, 실패 시 False 반환
    """
    try:
        s3.upload_file(
            Filename=filepath,
            Bucket=bucket,
            Key=access_key,
            ExtraArgs={
                "ACL": "public-read",
                "ContentType": "image/jpeg"    #Set appropriate content type as per the file
            }
        )
    except Exception as e:
        return False
    
    return True


def s3_get_image_url(s3, filename):
    """
    s3 : 연결된 s3 객체(boto3 client)
    filename : s3에 저장된 파일 명
    """
    location = s3.get_bucket_location(Bucket="takealook-bucket")["LocationConstraint"]
    return f"https://takealook-bucket.s3.ap-northeast-2.amazonaws.com/{filename}"