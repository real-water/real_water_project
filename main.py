import cv2
import numpy as np

def separate_ferrite_and_pearlite_graphite(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이진화 (Binary Thresholding)를 사용하여 밝은 영역(페라이트)와 어두운 영역(흑연 및 펄라이트)을 구분
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 밝은 영역(페라이트) 추출
    ferrite = cv2.bitwise_and(img, img, mask=binary_img)

    # 어두운 영역(흑연 및 펄라이트) 추출
    pearlite_graphite = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(binary_img))

    return ferrite, pearlite_graphite

# 이미지 경로를 입력하고 함수를 실행하세요
image_path = "C:/Users/soo/Desktop/shperoidal cast iron/image/X100-2.jpg"
ferrite, pearlite_graphite = separate_ferrite_and_pearlite_graphite(image_path)

# 결과를 확인하기 위해 이미지를 저장하세요
cv2.imwrite("ferrite.jpg", ferrite)
cv2.imwrite("pearlite_graphite.jpg", pearlite_graphite)
