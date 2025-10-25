#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🧩 Mediapipe 초기화 (Face Detection으로 변경)
# =========================================
# mp.solutions.hands 대신 mp.solutions.face_detection 사용
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# FaceDetection 모델 초기화
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, # 0은 짧은 범위 (가까운 얼굴), 1은 전체 범위 (먼 얼굴 포함)
    min_detection_confidence=0.5  # 탐지 신뢰도
)

# =========================================
# 📸 카메라 연결
# =========================================
# cap = cv2.VideoCapture(0)             # 기본 카메라 사용시
cap = cv2.VideoCapture("face.mp4")    # 동영상 파일 사용 시 (파일 이름 변경 권장)

print("📷 카메라 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. 카메라 연결을 확인하거나 동영상 파일을 확인하세요.")
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행 (hands.process 대신 face_detection.process 사용)
    result = face_detection.process(image_rgb)

    # 🧑‍ 얼굴 영역 표시
    if result.detections:
        # result.multi_hand_landmarks 대신 result.detections 사용
        for detection in result.detections:
            # 얼굴 영역(Bounding box) 및 6개의 특징점(랜드마크) 그리기
            mp_drawing.draw_detection(
                image, 
                detection,
                # Face Detection은 스타일을 별도로 제공하지 않고 기본값을 사용합니다.
            )
            
            # (선택 사항: 얼굴 감지 영역 위에 텍스트 추가 가능)
            # image_height, image_width, _ = image.shape
            # bbox_c = detection.location_data.relative_bounding_box
            # x_min = int(bbox_c.xmin * image_width)
            # y_min = int(bbox_c.ymin * image_height)
            # cv2.putText(image, f'{detection.score[0]*100:.2f}%', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # 화면 표시
    cv2.imshow('🧑‍ MediaPipe Face Detector', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()
face_detection.close() # 모델 객체 해제
