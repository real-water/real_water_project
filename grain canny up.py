import cv2
import numpy as np

def separate_ferrite_and_pearlite_graphite(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이진화 (Binary Thresholding)를 사용하여 밝은 영역(페라이트)와 어두운 영역(흑연 및 펄라이트)을 구분
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 밝은 영역(페라이트) 추출
    ferrite = cv2.bitwise_and(img, img, mask=binary_img)

    return ferrite

def emphasize_ferrite_edges(ferrite):
    # Gaussian blur를 적용하여 이미지의 노이즈를 줄임
    blurred_ferrite = cv2.GaussianBlur(ferrite, (3, 3), 0)

    # Canny 에지 검출 알고리즘의 임계값을 조정하여 경계 검출을 개선
    edges = cv2.Canny(blurred_ferrite, 50, 150)

    # 커널을 사용하여 경계선을 굵게 만듦 (크기와 반복 횟수를 조정)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)

    # 원본 페라이트 이미지와 에지를 합침 (가중치를 조절하여 경계선을 더욱 뚜렷하게 만듦)
    emphasized_edges = cv2.addWeighted(ferrite, 0.7, dilated_edges, 0.3, 0)

    return emphasized_edges

# 이미지 경로를 입력하고 함수를 실행하세요
image_path = "C:/Users/soo/Desktop/shperoidal cast iron/image/ferrite.jpg"
ferrite = separate_ferrite_and_pearlite_graphite(image_path)

# 페라이트 부분의 경계 강조
emphasized_ferrite_edges = emphasize_ferrite_edges(ferrite)

# 결과를 확인하기 위해 이미지를 저장하세요
cv2.imwrite("ferrite_emphasized_edges_new2.jpg", emphasized_ferrite_edges)
