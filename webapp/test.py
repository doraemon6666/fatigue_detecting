from flask import Flask, render_template, Response
import numpy as np
import dlib
import cv2
import imutils
import time
import math
from scipy.spatial import distance as dist
from imutils import face_utils
import os

app = Flask(__name__)

# 设置模型路径
model_path = os.path.join(os.path.dirname(__file__), 'model/shape_predictor_68_face_landmarks.dat')

# 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 世界坐标系(UVW)等参数省略，直接从上面的代码复制过来
# .... (object_pts, K, D, cam_matrix, dist_coeffs, reprojectsrc, line_pairs)

# 世界坐标系(UVW)定义 (假设人脸模型点的三维坐标)
object_pts = np.float32([
    [6.825897, 6.760612, 4.402142],    # 17 左眉左上角
    [1.330353, 7.122144, 6.903745],    # 21 右眉右上角
    [-1.330353, 7.122144, 6.903745],   # 22 右眉左上角
    [-6.825897, 6.760612, 4.402142],   # 26 右眉右上角
    [5.311432, 5.485328, 3.987654],    # 36 左眼左上角
    [1.789930, 5.393625, 4.413414],    # 39 左眼右上角
    [-1.789930, 5.393625, 4.413414],   # 42 右眼左上角
    [-5.311432, 5.485328, 3.987654],   # 45 右眼右上角
    [2.005628, 1.409845, 6.165652],    # 31 鼻子左侧
    [-2.005628, 1.409845, 6.165652],   # 35 鼻子右侧
    [2.774015, -2.080775, 5.048531],   # 48 嘴巴左侧
    [-2.774015, -2.080775, 5.048531],  # 54 嘴巴右侧
    [0.000000, -3.116408, 6.097667],   # 57 嘴巴中央
    [0.000000, -7.415691, 4.070434]    # 8 下巴尖
])

# 相机内参矩阵和畸变系数
cam_matrix = np.array([[6.5308391993466671e+02, 0.0, 3.1950000000000000e+02],
                       [0.0, 6.5308391993466671e+02, 2.3950000000000000e+02],
                       [0.0, 0.0, 1.0]])
dist_coeffs = np.zeros((4, 1)) # 畸变系数可以根据实际情况调整

# 投影到二维图像平面的三维模型点
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

# 定义常数
EYE_AR_THRESH = 0.35
EYE_AR_CONSEC_FRAMES = 3
EYE_CLOSED_SECONDS_THRESHOLD = 2
last_blink_time = time.time()
MAR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3
HAR_THRESH = 0.3
NOD_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0
mCOUNTER = 0
mTOTAL = 0
hCOUNTER = 0
hTOTAL = 0

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])
    B = np.linalg.norm(mouth[4] - mouth[7])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36], shape[39], shape[42], shape[45], shape[31], shape[35], shape[48], shape[54], shape[57], shape[8]])
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    return reprojectdst, euler_angle

def generate_frames():
    global COUNTER, TOTAL, mCOUNTER, mTOTAL, hCOUNTER, hTOTAL, last_blink_time
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=720)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                mar = mouth_aspect_ratio(mouth)
                reprojectdst, euler_angle = get_head_pose(shape)
                har = euler_angle[0, 0]

                eye_closed = ear < EYE_AR_THRESH
                if eye_closed:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        current_time = time.time()
                        time_since_last_blink = current_time - last_blink_time
                        if time_since_last_blink > EYE_CLOSED_SECONDS_THRESHOLD:
                            cv2.putText(frame, "SLEEP!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        last_blink_time = current_time
                else:
                    COUNTER = 0

                if mar > MAR_THRESH:
                    mCOUNTER += 1
                    cv2.putText(frame, "Yawning!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                        mTOTAL += 1
                    mCOUNTER = 0

                if har > HAR_THRESH:
                    hCOUNTER += 1
                else:
                    if hCOUNTER >= NOD_AR_CONSEC_FRAMES:
                        hTOTAL += 1
                    hCOUNTER = 0

                if TOTAL >= 50 or mTOTAL >= 15 or hTOTAL >= 15:
                    cv2.putText(frame, "SLEEP!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

