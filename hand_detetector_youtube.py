#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import yt_dlp # ydlp_opts에 오타가 있어 yt_dlp로 수정했습니다.

# =========================================
# 🎬 YouTube 스트림 URL 추출 함수
# =========================================
def get_youtube_stream_url():
    # YouTube 링크로부터 OpenCV가 재생 가능한 mp4 스트림 URL을 추출
    ydl_opts = {
        'quiet': True,             # 불필요한 로그 숨기기
        'noplaylist': True,        # 플레이리스트 재생 방지
        'format': 'best[ext=mp4]/best', # mp4 형식 우선 선택
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # 이 링크는 짧은 영상(shorts) 예시입니다.
        youtube_url = 'https://www.youtube.com/shorts/WMtG0TPT_XM?feature=share'
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_stream_url = info_dict.get('url', None)
        return video_stream_url

# =========================================
# 🧩 Mediapipe 초기화
# =========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,     # 동영상 입력
    max_num_hands=2,             # 최대 손 개수
    min_detection_confidence=0.5, # 탐지 신뢰도
    min_tracking_confidence=0.5  # 추적 신뢰도
)

# =========================================
# 🎥 YouTube 영상 연결
# =========================================
video_stream_url = get_youtube_stream_url()

if video_stream_url:
    print(f"🎬 YouTube 스트림 링크를 가져왔습니다.")
    # OpenCV를 사용하여 YouTube 스트림 열기
    cap = cv2.VideoCapture(video_stream_url)
else:
    print("❌ YouTube 영상을 찾을 수 없습니다.")
    exit()

# =========================================
# 💻 메인 루프
# =========================================
if not cap.isOpened():
    print("❌ YouTube 영상을 열 수 없습니다.")
    exit()

print("🎥 YouTube 스트림 재생 시작 - ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 영상 스트림이 끝났거나 프레임을 읽지 못했습니다.")
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 손 검출 수행
    result = hands.process(image_rgb)

    # 🖐️ 손 랜드마크 표시
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    # 결과 화면 표시
    cv2.imshow('🖐️ MediaPipe Hand Detector (YouTube)', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()

