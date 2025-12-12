import cv2
import mediapipe as mp
import numpy as np

# 角度を計算する関数
def calculate_angle(a, b, c):
    a = np.array(a) # 耳
    b = np.array(b) # 肩
    c = np.array(c) # 腰

    # ラジアンから度数へ変換
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# 2点間の距離を計算する関数（正規化された座標を使用）
def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

# MediaPipeの準備
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Webカメラのキャプチャ開始
cap = cv2.VideoCapture(1)

# 姿勢推定のセットアップ
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 画像をRGBに変換（MediaPipe用）
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # 推論実行
        results = pose.process(image)
    
        # 画像をBGRに戻す（OpenCV表示用）
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 画面サイズ取得
        h, w, _ = image.shape

        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- 正面からの判定ロジック ---
            # 左右の耳と肩の座標を取得
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            
            # 鼻の座標（顔の大きさの基準として使用）
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            # 距離計算（Y軸方向の距離＝高さの差を見るのが効果的）
            # 猫背で顔が前に出ると、2D画像上では耳と肩のY座標が近づく、または耳が肩より下がることはないが距離が縮まる
            left_dist = abs(left_shoulder[1] - left_ear[1])
            right_dist = abs(right_shoulder[1] - right_ear[1])
            avg_dist = (left_dist + right_dist) / 2

            # 基準となる距離（例えば肩幅など）で正規化するとカメラ距離に依存しにくくなりますが、
            # ここでは簡易的に「肩と耳の距離」が一定以下になったら猫背と判定します。
            # ※カメラとの距離で閾値が変わるため、本来は「肩幅に対する比率」などを使うとより堅牢です。
            
            # 肩幅を計算（正規化の基準）
            shoulder_width = calculate_distance(left_shoulder, right_shoulder)
            
            # 比率 = (耳と肩の垂直距離) / 肩幅
            # 姿勢が良いと首が伸びて比率が大きくなり、猫背（顔が前、肩が上がる）だと比率が小さくなる
            ratio = avg_dist / shoulder_width

            # 閾値設定（環境に合わせて調整してください）
            THRESHOLD_RATIO = 0.7 

            if ratio < THRESHOLD_RATIO:
                stage = "Bad Posture"
                color = (0, 0, 255) # 赤色
            else:
                stage = "Good Posture"
                color = (0, 255, 0) # 緑色
                
            # 画面への情報描画
            # 左肩付近に数値を表示
            display_pos = tuple(np.multiply(left_shoulder, [w, h]).astype(int))
            cv2.putText(image, f"Ratio: {ratio:.2f}", 
                           display_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, stage, 
                        (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        except:
            pass

        # 骨格の描画
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Nekoze Detector', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()