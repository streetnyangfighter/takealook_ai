# -*- coding: utf-8 -*-

# 이진화 한 이미지 두 개 비교하여 점수 산출
def scoring(img1, img2):
    count = 0
    for i in range(1100):
        for j in range(1100):
            if img1[i][j] == img2[i][j]:
                count += 1
            
    return (count / (1100*1100)) * 100