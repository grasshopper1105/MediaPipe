import csv
import os
import cv2
import cv2 as cv
import pandas as pd
import mediapipe as mp
import numpy as np
import torch
from numpy import float32
from LSTMTest import LSTM


# Function to create CSV file or add dataset to the existed CSV file
def toCSV_detect(filecsv,
                 left_shoulderX, left_shoulderY,
                 right_shoulderX, right_shoulderY,
                 left_elbowX, left_elbowY,
                 right_elbowX, right_elbowY,
                 left_wristX, left_wristY,
                 left_thumb_CmcX, left_thumb_CmcY,
                 left_thumb_McpX, left_thumb_McpY,
                 left_thumb_IpX, left_thumb_IpY,
                 left_thumb_TipX, left_thumb_TipY,
                 left_index_McpX, left_index_McpY,
                 left_index_PipX, left_index_PipY,
                 left_index_DipX, left_index_DipY,
                 left_index_TipX, left_index_TipY,
                 left_middle_McpX, left_middle_McpY,
                 left_middle_PipX, left_middle_PipY,
                 left_middle_DipX, left_middle_DipY,
                 left_middle_TipX, left_middle_TipY,
                 left_ring_McpX, left_ring_McpY,
                 left_ring_PipX, left_ring_PipY,
                 left_ring_DipX, left_ring_DipY,
                 left_ring_TipX, left_ring_TipY,
                 left_pinky_McpX, left_pinky_McpY,
                 left_pinky_PipX, left_pinky_PipY,
                 left_pinky_DipX, left_pinky_DipY,
                 left_pinky_TipX, left_pinky_TipY,
                 right_wristX, right_wristY,
                 right_thumb_CmcX, right_thumb_CmcY,
                 right_thumb_McpX, right_thumb_McpY,
                 right_thumb_IpX, right_thumb_IpY,
                 right_thumb_TipX, right_thumb_TipY,
                 right_index_McpX, right_index_McpY,
                 right_index_PipX, right_index_PipY,
                 right_index_DipX, right_index_DipY,
                 right_index_TipX, right_index_TipY,
                 right_middle_McpX, right_middle_McpY,
                 right_middle_PipX, right_middle_PipY,
                 right_middle_DipX, right_middle_DipY,
                 right_middle_TipX, right_middle_TipY,
                 right_ring_McpX, right_ring_McpY,
                 right_ring_PipX, right_ring_PipY,
                 right_ring_DipX, right_ring_DipY,
                 right_ring_TipX, right_ring_TipY,
                 right_pinky_McpX, right_pinky_McpY,
                 right_pinky_PipX, right_pinky_PipY,
                 right_pinky_DipX, right_pinky_DipY,
                 right_pinky_TipX, right_pinky_TipY):
    if os.path.isfile(filecsv):
        # print ("File exist thus shall write append to the file")
        with open(filecsv, 'a+', newline='') as file:
            # Create a writer object from csv module
            writer = csv.writer(file)
            writer.writerow([left_shoulderX, left_shoulderY,
                             right_shoulderX, right_shoulderY,
                             left_elbowX, left_elbowY,
                             right_elbowX, right_elbowY,
                             left_wristX, left_wristY,
                             left_thumb_CmcX, left_thumb_CmcY,
                             left_thumb_McpX, left_thumb_McpY,
                             left_thumb_IpX, left_thumb_IpY,
                             left_thumb_TipX, left_thumb_TipY,
                             left_index_McpX, left_index_McpY,
                             left_index_PipX, left_index_PipY,
                             left_index_DipX, left_index_DipY,
                             left_index_TipX, left_index_TipY,
                             left_middle_McpX, left_middle_McpY,
                             left_middle_PipX, left_middle_PipY,
                             left_middle_DipX, left_middle_DipY,
                             left_middle_TipX, left_middle_TipY,
                             left_ring_McpX, left_ring_McpY,
                             left_ring_PipX, left_ring_PipY,
                             left_ring_DipX, left_ring_DipY,
                             left_ring_TipX, left_ring_TipY,
                             left_pinky_McpX, left_pinky_McpY,
                             left_pinky_PipX, left_pinky_PipY,
                             left_pinky_DipX, left_pinky_DipY,
                             left_pinky_TipX, left_pinky_TipY,
                             right_wristX, right_wristY,
                             right_thumb_CmcX, right_thumb_CmcY,
                             right_thumb_McpX, right_thumb_McpY,
                             right_thumb_IpX, right_thumb_IpY,
                             right_thumb_TipX, right_thumb_TipY,
                             right_index_McpX, right_index_McpY,
                             right_index_PipX, right_index_PipY,
                             right_index_DipX, right_index_DipY,
                             right_index_TipX, right_index_TipY,
                             right_middle_McpX, right_middle_McpY,
                             right_middle_PipX, right_middle_PipY,
                             right_middle_DipX, right_middle_DipY,
                             right_middle_TipX, right_middle_TipY,
                             right_ring_McpX, right_ring_McpY,
                             right_ring_PipX, right_ring_PipY,
                             right_ring_DipX, right_ring_DipY,
                             right_ring_TipX, right_ring_TipY,
                             right_pinky_McpX, right_pinky_McpY,
                             right_pinky_PipX, right_pinky_PipY,
                             right_pinky_DipX, right_pinky_DipY,
                             right_pinky_TipX, right_pinky_TipY])
    else:
        # print ("File not exist thus shall create new file as", filecsv)
        with open(filecsv, 'w', newline='') as file:
            # Create a writer object from csv module
            writer = csv.writer(file)
            writer.writerow(['left_shoulderX', 'left_shoulderY',
                             'right_shoulderX', 'right_shoulderY',
                             'left_elbowX', 'left_elbowY',
                             'right_elbowX', 'right_elbowY',
                             'left_wristX', 'left_wristY',
                             'left_thumb_CmcX', 'left_thumb_CmcY',
                             'left_thumb_McpX', 'left_thumb_McpY',
                             'left_thumb_IpX', 'left_thumb_IpY',
                             'left_thumb_TipX', 'left_thumb_TipY',
                             'left_index_McpX', 'left_index_McpY',
                             'left_index_PipX', 'left_index_PipY',
                             'left_index_DipX', 'left_index_DipY',
                             'left_index_TipX', 'left_index_TipY',
                             'left_middle_McpX', 'left_middle_McpY',
                             'left_middle_PipX', 'left_middle_PipY',
                             'left_middle_DipX', 'left_middle_DipY',
                             'left_middle_TipX', 'left_middle_TipY',
                             'left_ring_McpX', 'left_ring_McpY',
                             'left_ring_PipX', 'left_ring_PipY',
                             'left_ring_DipX', 'left_ring_DipY',
                             'left_ring_TipX', 'left_ring_TipY',
                             'left_pinky_McpX', 'left_pinky_McpY',
                             'left_pinky_PipX', 'left_pinky_PipY',
                             'left_pinky_DipX', 'left_pinky_DipY',
                             'left_pinky_TipX', 'left_pinky_TipY',
                             'right_wristX', 'right_wristY',
                             'right_thumb_CmcX', 'right_thumb_CmcY',
                             'right_thumb_McpX', 'right_thumb_McpY',
                             'right_thumb_IpX', 'right_thumb_IpY',
                             'right_thumb_TipX', 'right_thumb_TipY',
                             'right_index_McpX', 'right_index_McpY',
                             'right_index_PipX', 'right_index_PipY',
                             'right_index_DipX', 'right_index_DipY',
                             'right_index_TipX', 'right_index_TipY',
                             'right_middle_McpX', 'right_middle_McpY',
                             'right_middle_PipX', 'right_middle_PipY',
                             'right_middle_DipX', 'right_middle_DipY',
                             'right_middle_TipX', 'right_middle_TipY',
                             'right_ring_McpX', 'right_ring_McpY',
                             'right_ring_PipX', 'right_ring_PipY',
                             'right_ring_DipX', 'right_ring_DipY',
                             'right_ring_TipX', 'right_ring_TipY',
                             'right_pinky_McpX', 'right_pinky_McpY',
                             'right_pinky_PipX', 'right_pinky_PipY',
                             'right_pinky_DipX', 'right_pinky_DipY',
                             'right_pinky_TipX', 'right_pinky_TipY'])
            writer.writerow([left_shoulderX, left_shoulderY,
                             right_shoulderX, right_shoulderY,
                             left_elbowX, left_elbowY,
                             right_elbowX, right_elbowY,
                             left_wristX, left_wristY,
                             left_thumb_CmcX, left_thumb_CmcY,
                             left_thumb_McpX, left_thumb_McpY,
                             left_thumb_IpX, left_thumb_IpY,
                             left_thumb_TipX, left_thumb_TipY,
                             left_index_McpX, left_index_McpY,
                             left_index_PipX, left_index_PipY,
                             left_index_DipX, left_index_DipY,
                             left_index_TipX, left_index_TipY,
                             left_middle_McpX, left_middle_McpY,
                             left_middle_PipX, left_middle_PipY,
                             left_middle_DipX, left_middle_DipY,
                             left_middle_TipX, left_middle_TipY,
                             left_ring_McpX, left_ring_McpY,
                             left_ring_PipX, left_ring_PipY,
                             left_ring_DipX, left_ring_DipY,
                             left_ring_TipX, left_ring_TipY,
                             left_pinky_McpX, left_pinky_McpY,
                             left_pinky_PipX, left_pinky_PipY,
                             left_pinky_DipX, left_pinky_DipY,
                             left_pinky_TipX, left_pinky_TipY,
                             right_wristX, right_wristY,
                             right_thumb_CmcX, right_thumb_CmcY,
                             right_thumb_McpX, right_thumb_McpY,
                             right_thumb_IpX, right_thumb_IpY,
                             right_thumb_TipX, right_thumb_TipY,
                             right_index_McpX, right_index_McpY,
                             right_index_PipX, right_index_PipY,
                             right_index_DipX, right_index_DipY,
                             right_index_TipX, right_index_TipY,
                             right_middle_McpX, right_middle_McpY,
                             right_middle_PipX, right_middle_PipY,
                             right_middle_DipX, right_middle_DipY,
                             right_middle_TipX, right_middle_TipY,
                             right_ring_McpX, right_ring_McpY,
                             right_ring_PipX, right_ring_PipY,
                             right_ring_DipX, right_ring_DipY,
                             right_ring_TipX, right_ring_TipY,
                             right_pinky_McpX, right_pinky_McpY,
                             right_pinky_PipX, right_pinky_PipY,
                             right_pinky_DipX, right_pinky_DipY,
                             right_pinky_TipX, right_pinky_TipY])


def calculate_files_num(_path, _class):
    _sum = 0
    _file_sum_list = []
    for _ in range(0, _class):
        path = _path + str(_).rjust(3, '0')
        files = os.listdir(path)  # 读入文件夹
        _num = len(files)  # 统计文件夹中的文件个数
        _sum += _num
        _file_sum_list.append(_sum)
    return _sum, _file_sum_list


# 超参数
FRAME = 20  # 截取帧数
NUM_CLASS = 70  # 手语词类
BATCH_SIZE = 4  # batch大小
HIDDEN_SIZE = 256  # 隐藏层
INPUT_SIZE = 92  # 特征数
DROP_OUT = 0.1  # 随机抛弃
NUM_LAYER = 2  # RNN层数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYER, num_classes=NUM_CLASS,
             drop_p=DROP_OUT, batch_size=BATCH_SIZE).to(device)
model.load_state_dict(torch.load("/root/hy-tmp/biLSTM_20frame_98acc.mdl"))  # 远程
# model.load_state_dict(torch.load("D:\\Software\\PyCharm\\AI\\MediaPipe\\best.mdl"))
model.eval()


def extract_feature(image):
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(static_image_mode=True, model_complexity=1, min_tracking_confidence=0.1) as holistic:
        results = holistic.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        if results.pose_landmarks:

            noseX = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x
            noseY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y

            left_shoulderX = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x - noseX
            left_shoulderY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y - noseY

            right_shoulderX = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x - noseX
            right_shoulderY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y - noseY

            left_elbowX = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x - noseX
            left_elbowY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y - noseY

            right_elbowX = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x - noseX
            right_elbowY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y - noseY

            features_pose = [left_shoulderX, left_shoulderY,
                             right_shoulderX, right_shoulderY,
                             left_elbowX, left_elbowY,
                             right_elbowX, right_elbowY]

        else:

            noseX = np.nan
            noseY = np.nan

            left_shoulderX = np.nan
            left_shoulderY = np.nan

            right_shoulderX = np.nan
            right_shoulderY = np.nan

            left_elbowX = np.nan
            left_elbowY = np.nan

            right_elbowX = np.nan
            right_elbowY = np.nan

            features_pose = [left_shoulderX, left_shoulderY,
                             right_shoulderX, right_shoulderY,
                             left_elbowX, left_elbowY,
                             right_elbowX, right_elbowY]

        if results.left_hand_landmarks:

            left_wristX = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x - noseX
            left_wristY = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y - noseY

            left_thumb_CmcX = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].x - noseX
            left_thumb_CmcY = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].y - noseY

            left_thumb_McpX = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].x - noseX
            left_thumb_McpY = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].y - noseY

            left_thumb_IpX = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].x - noseX
            left_thumb_IpY = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].y - noseY

            left_thumb_TipX = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x - noseX
            left_thumb_TipY = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y - noseY

            left_index_McpX = results.left_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.INDEX_FINGER_MCP].x - noseX
            left_index_McpY = results.left_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.INDEX_FINGER_MCP].y - noseY

            left_index_PipX = results.left_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.INDEX_FINGER_PIP].x - noseX
            left_index_PipY = results.left_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.INDEX_FINGER_PIP].y - noseY

            left_index_DipX = results.left_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.INDEX_FINGER_DIP].x - noseX
            left_index_DipY = results.left_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.INDEX_FINGER_DIP].y - noseY

            left_index_TipX = results.left_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.INDEX_FINGER_TIP].x - noseX
            left_index_TipY = results.left_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.INDEX_FINGER_TIP].y - noseY

            left_middle_McpX = results.left_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x - noseX
            left_middle_McpY = results.left_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y - noseY

            left_middle_PipX = results.left_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x - noseX
            left_middle_PipY = results.left_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y - noseY

            left_middle_DipX = results.left_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x - noseX
            left_middle_DipY = results.left_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y - noseY

            left_middle_TipX = results.left_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x - noseX
            left_middle_TipY = results.left_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y - noseY

            left_ring_McpX = results.left_hand_landmarks.landmark[
                                 mp_holistic.HandLandmark.RING_FINGER_MCP].x - noseX
            left_ring_McpY = results.left_hand_landmarks.landmark[
                                 mp_holistic.HandLandmark.RING_FINGER_MCP].y - noseY

            left_ring_PipX = results.left_hand_landmarks.landmark[
                                 mp_holistic.HandLandmark.RING_FINGER_PIP].x - noseX
            left_ring_PipY = results.left_hand_landmarks.landmark[
                                 mp_holistic.HandLandmark.RING_FINGER_PIP].y - noseY

            left_ring_DipX = results.left_hand_landmarks.landmark[
                                 mp_holistic.HandLandmark.RING_FINGER_DIP].x - noseX
            left_ring_DipY = results.left_hand_landmarks.landmark[
                                 mp_holistic.HandLandmark.RING_FINGER_DIP].y - noseY

            left_ring_TipX = results.left_hand_landmarks.landmark[
                                 mp_holistic.HandLandmark.RING_FINGER_TIP].x - noseX
            left_ring_TipY = results.left_hand_landmarks.landmark[
                                 mp_holistic.HandLandmark.RING_FINGER_TIP].y - noseY

            left_pinky_McpX = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].x - noseX
            left_pinky_McpY = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].y - noseY

            left_pinky_PipX = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].x - noseX
            left_pinky_PipY = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].y - noseY

            left_pinky_DipX = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].x - noseX
            left_pinky_DipY = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].y - noseY

            left_pinky_TipX = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x - noseX
            left_pinky_TipY = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y - noseY

            features_left_hands = [left_wristX, left_wristY,
                                   left_thumb_CmcX, left_thumb_CmcY,
                                   left_thumb_McpX, left_thumb_McpY,
                                   left_thumb_IpX, left_thumb_IpY,
                                   left_thumb_TipX, left_thumb_TipY,
                                   left_index_McpX, left_index_McpY,
                                   left_index_PipX, left_index_PipY,
                                   left_index_DipX, left_index_DipY,
                                   left_index_TipX, left_index_TipY,
                                   left_middle_McpX, left_middle_McpY,
                                   left_middle_PipX, left_middle_PipY,
                                   left_middle_DipX, left_middle_DipY,
                                   left_middle_TipX, left_middle_TipY,
                                   left_ring_McpX, left_ring_McpY,
                                   left_ring_PipX, left_ring_PipY,
                                   left_ring_DipX, left_ring_DipY,
                                   left_ring_TipX, left_ring_TipY,
                                   left_pinky_McpX, left_pinky_McpY,
                                   left_pinky_PipX, left_pinky_PipY,
                                   left_pinky_DipX, left_pinky_DipY,
                                   left_pinky_TipX, left_pinky_TipY]

        else:

            left_wristX = np.nan
            left_wristY = np.nan

            left_thumb_CmcX = np.nan
            left_thumb_CmcY = np.nan

            left_thumb_McpX = np.nan
            left_thumb_McpY = np.nan

            left_thumb_IpX = np.nan
            left_thumb_IpY = np.nan

            left_thumb_TipX = np.nan
            left_thumb_TipY = np.nan

            left_index_McpX = np.nan
            left_index_McpY = np.nan

            left_index_PipX = np.nan
            left_index_PipY = np.nan

            left_index_DipX = np.nan
            left_index_DipY = np.nan

            left_index_TipX = np.nan
            left_index_TipY = np.nan

            left_middle_McpX = np.nan
            left_middle_McpY = np.nan

            left_middle_PipX = np.nan
            left_middle_PipY = np.nan

            left_middle_DipX = np.nan
            left_middle_DipY = np.nan

            left_middle_TipX = np.nan
            left_middle_TipY = np.nan

            left_ring_McpX = np.nan
            left_ring_McpY = np.nan

            left_ring_PipX = np.nan
            left_ring_PipY = np.nan

            left_ring_DipX = np.nan
            left_ring_DipY = np.nan

            left_ring_TipX = np.nan
            left_ring_TipY = np.nan

            left_pinky_McpX = np.nan
            left_pinky_McpY = np.nan

            left_pinky_PipX = np.nan
            left_pinky_PipY = np.nan

            left_pinky_DipX = np.nan
            left_pinky_DipY = np.nan

            left_pinky_TipX = np.nan
            left_pinky_TipY = np.nan

            features_left_hands = [left_wristX, left_wristY,
                                   left_thumb_CmcX, left_thumb_CmcY,
                                   left_thumb_McpX, left_thumb_McpY,
                                   left_thumb_IpX, left_thumb_IpY,
                                   left_thumb_TipX, left_thumb_TipY,
                                   left_index_McpX, left_index_McpY,
                                   left_index_PipX, left_index_PipY,
                                   left_index_DipX, left_index_DipY,
                                   left_index_TipX, left_index_TipY,
                                   left_middle_McpX, left_middle_McpY,
                                   left_middle_PipX, left_middle_PipY,
                                   left_middle_DipX, left_middle_DipY,
                                   left_middle_TipX, left_middle_TipY,
                                   left_ring_McpX, left_ring_McpY,
                                   left_ring_PipX, left_ring_PipY,
                                   left_ring_DipX, left_ring_DipY,
                                   left_ring_TipX, left_ring_TipY,
                                   left_pinky_McpX, left_pinky_McpY,
                                   left_pinky_PipX, left_pinky_PipY,
                                   left_pinky_DipX, left_pinky_DipY,
                                   left_pinky_TipX, left_pinky_TipY]

        if results.right_hand_landmarks:

            right_wristX = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x - noseX
            right_wristY = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y - noseY

            right_thumb_CmcX = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].x - noseX
            right_thumb_CmcY = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.THUMB_CMC].y - noseY

            right_thumb_McpX = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].x - noseX
            right_thumb_McpY = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.THUMB_MCP].y - noseY

            right_thumb_IpX = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].x - noseX
            right_thumb_IpY = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].y - noseY

            right_thumb_TipX = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x - noseX
            right_thumb_TipY = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.THUMB_TIP].y - noseY

            right_index_McpX = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.INDEX_FINGER_MCP].x - noseX
            right_index_McpY = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.INDEX_FINGER_MCP].y - noseY

            right_index_PipX = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.INDEX_FINGER_PIP].x - noseX
            right_index_PipY = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.INDEX_FINGER_PIP].y - noseY

            right_index_DipX = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.INDEX_FINGER_DIP].x - noseX
            right_index_DipY = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.INDEX_FINGER_DIP].y - noseY

            right_index_TipX = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.INDEX_FINGER_TIP].x - noseX
            right_index_TipY = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.INDEX_FINGER_TIP].y - noseY

            right_middle_McpX = results.right_hand_landmarks.landmark[
                                    mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x - noseX
            right_middle_McpY = results.right_hand_landmarks.landmark[
                                    mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y - noseY

            right_middle_PipX = results.right_hand_landmarks.landmark[
                                    mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x - noseX
            right_middle_PipY = results.right_hand_landmarks.landmark[
                                    mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y - noseY

            right_middle_DipX = results.right_hand_landmarks.landmark[
                                    mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x - noseX
            right_middle_DipY = results.right_hand_landmarks.landmark[
                                    mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y - noseY

            right_middle_TipX = results.right_hand_landmarks.landmark[
                                    mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x - noseX
            right_middle_TipY = results.right_hand_landmarks.landmark[
                                    mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y - noseY

            right_ring_McpX = results.right_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.RING_FINGER_MCP].x - noseX
            right_ring_McpY = results.right_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.RING_FINGER_MCP].y - noseY

            right_ring_PipX = results.right_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.RING_FINGER_PIP].x - noseX
            right_ring_PipY = results.right_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.RING_FINGER_PIP].y - noseY

            right_ring_DipX = results.right_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.RING_FINGER_DIP].x - noseX
            right_ring_DipY = results.right_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.RING_FINGER_DIP].y - noseY

            right_ring_TipX = results.right_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.RING_FINGER_TIP].x - noseX
            right_ring_TipY = results.right_hand_landmarks.landmark[
                                  mp_holistic.HandLandmark.RING_FINGER_TIP].y - noseY

            right_pinky_McpX = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].x - noseX
            right_pinky_McpY = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.PINKY_MCP].y - noseY

            right_pinky_PipX = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].x - noseX
            right_pinky_PipY = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.PINKY_PIP].y - noseY

            right_pinky_DipX = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].x - noseX
            right_pinky_DipY = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.PINKY_DIP].y - noseY

            right_pinky_TipX = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x - noseX
            right_pinky_TipY = results.right_hand_landmarks.landmark[
                                   mp_holistic.HandLandmark.PINKY_TIP].y - noseY

            features_right_hands = [right_wristX, right_wristY,
                                    right_thumb_CmcX, right_thumb_CmcY,
                                    right_thumb_McpX, right_thumb_McpY,
                                    right_thumb_IpX, right_thumb_IpY,
                                    right_thumb_TipX, right_thumb_TipY,
                                    right_index_McpX, right_index_McpY,
                                    right_index_PipX, right_index_PipY,
                                    right_index_DipX, right_index_DipY,
                                    right_index_TipX, right_index_TipY,
                                    right_middle_McpX, right_middle_McpY,
                                    right_middle_PipX, right_middle_PipY,
                                    right_middle_DipX, right_middle_DipY,
                                    right_middle_TipX, right_middle_TipY,
                                    right_ring_McpX, right_ring_McpY,
                                    right_ring_PipX, right_ring_PipY,
                                    right_ring_DipX, right_ring_DipY,
                                    right_ring_TipX, right_ring_TipY,
                                    right_pinky_McpX, right_pinky_McpY,
                                    right_pinky_PipX, right_pinky_PipY,
                                    right_pinky_DipX, right_pinky_DipY,
                                    right_pinky_TipX, right_pinky_TipY]

        else:

            right_wristX = np.nan
            right_wristY = np.nan

            right_thumb_CmcX = np.nan
            right_thumb_CmcY = np.nan

            right_thumb_McpX = np.nan
            right_thumb_McpY = np.nan

            right_thumb_IpX = np.nan
            right_thumb_IpY = np.nan

            right_thumb_TipX = np.nan
            right_thumb_TipY = np.nan

            right_index_McpX = np.nan
            right_index_McpY = np.nan

            right_index_PipX = np.nan
            right_index_PipY = np.nan

            right_index_DipX = np.nan
            right_index_DipY = np.nan

            right_index_TipX = np.nan
            right_index_TipY = np.nan

            right_middle_McpX = np.nan
            right_middle_McpY = np.nan

            right_middle_PipX = np.nan
            right_middle_PipY = np.nan

            right_middle_DipX = np.nan
            right_middle_DipY = np.nan

            right_middle_TipX = np.nan
            right_middle_TipY = np.nan

            right_ring_McpX = np.nan
            right_ring_McpY = np.nan

            right_ring_PipX = np.nan
            right_ring_PipY = np.nan

            right_ring_DipX = np.nan
            right_ring_DipY = np.nan

            right_ring_TipX = np.nan
            right_ring_TipY = np.nan

            right_pinky_McpX = np.nan
            right_pinky_McpY = np.nan

            right_pinky_PipX = np.nan
            right_pinky_PipY = np.nan

            right_pinky_DipX = np.nan
            right_pinky_DipY = np.nan

            right_pinky_TipX = np.nan
            right_pinky_TipY = np.nan

            features_right_hands = [right_wristX, right_wristY,
                                    right_thumb_CmcX, right_thumb_CmcY,
                                    right_thumb_McpX, right_thumb_McpY,
                                    right_thumb_IpX, right_thumb_IpY,
                                    right_thumb_TipX, right_thumb_TipY,
                                    right_index_McpX, right_index_McpY,
                                    right_index_PipX, right_index_PipY,
                                    right_index_DipX, right_index_DipY,
                                    right_index_TipX, right_index_TipY,
                                    right_middle_McpX, right_middle_McpY,
                                    right_middle_PipX, right_middle_PipY,
                                    right_middle_DipX, right_middle_DipY,
                                    right_middle_TipX, right_middle_TipY,
                                    right_ring_McpX, right_ring_McpY,
                                    right_ring_PipX, right_ring_PipY,
                                    right_ring_DipX, right_ring_DipY,
                                    right_ring_TipX, right_ring_TipY,
                                    right_pinky_McpX, right_pinky_McpY,
                                    right_pinky_PipX, right_pinky_PipY,
                                    right_pinky_DipX, right_pinky_DipY,
                                    right_pinky_TipX, right_pinky_TipY]

        features = features_pose + features_left_hands + features_right_hands
        return features


# cap = cv2.VideoCapture('D:\\Software\\PyCharm\\AI\\MediaPipe\\pic\\004.avi')
cap = cv2.VideoCapture('/root/hy-tmp/pic/004.avi')
PROCESS_PATH = 'temp.csv'
if os.path.exists(PROCESS_PATH):
    os.remove(PROCESS_PATH)

FRAME_CNT = 0  # 记录读取多少帧
SAVE_CNT = 0
cnt = 1  # 记录总帧数
STEP = 3  # 每frameFrequency保存一张图片

pics = []
features = []
while cap.isOpened():
    flag, frame = cap.read()  # read方法 读取每一张 flag是否读取成功 frame 读取内容
    if FRAME_CNT == 0:
        FRAME_CNT += 1
        continue
    if FRAME_CNT % STEP == 0:
        feature_list = [left_shoulderX, left_shoulderY,
                        right_shoulderX, right_shoulderY,
                        left_elbowX, left_elbowY,
                        right_elbowX, right_elbowY,
                        left_wristX, left_wristY,
                        left_thumb_CmcX, left_thumb_CmcY,
                        left_thumb_McpX, left_thumb_McpY,
                        left_thumb_IpX, left_thumb_IpY,
                        left_thumb_TipX, left_thumb_TipY,
                        left_index_McpX, left_index_McpY,
                        left_index_PipX, left_index_PipY,
                        left_index_DipX, left_index_DipY,
                        left_index_TipX, left_index_TipY,
                        left_middle_McpX, left_middle_McpY,
                        left_middle_PipX, left_middle_PipY,
                        left_middle_DipX, left_middle_DipY,
                        left_middle_TipX, left_middle_TipY,
                        left_ring_McpX, left_ring_McpY,
                        left_ring_PipX, left_ring_PipY,
                        left_ring_DipX, left_ring_DipY,
                        left_ring_TipX, left_ring_TipY,
                        left_pinky_McpX, left_pinky_McpY,
                        left_pinky_PipX, left_pinky_PipY,
                        left_pinky_DipX, left_pinky_DipY,
                        left_pinky_TipX, left_pinky_TipY,
                        right_wristX, right_wristY,
                        right_thumb_CmcX, right_thumb_CmcY,
                        right_thumb_McpX, right_thumb_McpY,
                        right_thumb_IpX, right_thumb_IpY,
                        right_thumb_TipX, right_thumb_TipY,
                        right_index_McpX, right_index_McpY,
                        right_index_PipX, right_index_PipY,
                        right_index_DipX, right_index_DipY,
                        right_index_TipX, right_index_TipY,
                        right_middle_McpX, right_middle_McpY,
                        right_middle_PipX, right_middle_PipY,
                        right_middle_DipX, right_middle_DipY,
                        right_middle_TipX, right_middle_TipY,
                        right_ring_McpX, right_ring_McpY,
                        right_ring_PipX, right_ring_PipY,
                        right_ring_DipX, right_ring_DipY,
                        right_ring_TipX, right_ring_TipY,
                        right_pinky_McpX, right_pinky_McpY,
                        right_pinky_PipX, right_pinky_PipY,
                        right_pinky_DipX, right_pinky_DipY,
                        right_pinky_TipX, right_pinky_TipY] = extract_feature(frame)
        if (left_shoulderX != np.nan) and (left_wristX != np.nan):
            toCSV_detect(PROCESS_PATH,
                         left_shoulderX, left_shoulderY,
                         right_shoulderX, right_shoulderY,
                         left_elbowX, left_elbowY,
                         right_elbowX, right_elbowY,
                         left_wristX, left_wristY,
                         left_thumb_CmcX, left_thumb_CmcY,
                         left_thumb_McpX, left_thumb_McpY,
                         left_thumb_IpX, left_thumb_IpY,
                         left_thumb_TipX, left_thumb_TipY,
                         left_index_McpX, left_index_McpY,
                         left_index_PipX, left_index_PipY,
                         left_index_DipX, left_index_DipY,
                         left_index_TipX, left_index_TipY,
                         left_middle_McpX, left_middle_McpY,
                         left_middle_PipX, left_middle_PipY,
                         left_middle_DipX, left_middle_DipY,
                         left_middle_TipX, left_middle_TipY,
                         left_ring_McpX, left_ring_McpY,
                         left_ring_PipX, left_ring_PipY,
                         left_ring_DipX, left_ring_DipY,
                         left_ring_TipX, left_ring_TipY,
                         left_pinky_McpX, left_pinky_McpY,
                         left_pinky_PipX, left_pinky_PipY,
                         left_pinky_DipX, left_pinky_DipY,
                         left_pinky_TipX, left_pinky_TipY,
                         right_wristX, right_wristY,
                         right_thumb_CmcX, right_thumb_CmcY,
                         right_thumb_McpX, right_thumb_McpY,
                         right_thumb_IpX, right_thumb_IpY,
                         right_thumb_TipX, right_thumb_TipY,
                         right_index_McpX, right_index_McpY,
                         right_index_PipX, right_index_PipY,
                         right_index_DipX, right_index_DipY,
                         right_index_TipX, right_index_TipY,
                         right_middle_McpX, right_middle_McpY,
                         right_middle_PipX, right_middle_PipY,
                         right_middle_DipX, right_middle_DipY,
                         right_middle_TipX, right_middle_TipY,
                         right_ring_McpX, right_ring_McpY,
                         right_ring_PipX, right_ring_PipY,
                         right_ring_DipX, right_ring_DipY,
                         right_ring_TipX, right_ring_TipY,
                         right_pinky_McpX, right_pinky_McpY,
                         right_pinky_PipX, right_pinky_PipY,
                         right_pinky_DipX, right_pinky_DipY,
                         right_pinky_TipX, right_pinky_TipY)
        SAVE_CNT += 1
        if SAVE_CNT == FRAME:
            break
    FRAME_CNT += 1

data = pd.read_csv(PROCESS_PATH, header=0, sep=',')
for j in range(data.shape[1]):
    data.iloc[:, j].interpolate(method='spline', order=3, limit_direction='both', inplace=True)
data.fillna(method='pad', axis=0, inplace=True)
print(data.shape)

with torch.no_grad():
    final_features = torch.from_numpy(np.array(data).astype(float32)).unsqueeze(0).to(device)
    res = (model(final_features)[0].cpu().numpy().squeeze())
    idx = res.argmax(0)
    print(idx)
