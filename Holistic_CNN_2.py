# Import the much needed stuff for training
import csv
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
from  MediaPipe.tools.data_to_csv import toCSV

np.set_printoptions(precision=3, suppress=True)  # 禁用科学计数法，设置小数保留后三位


# Function to Extract Feature from images or Frame
def extract_feature(input_image):
    mp_holistic = mp.solutions.holistic

    image = cv.imread(input_image)
    with mp_holistic.Holistic(static_image_mode=True, model_complexity=1, min_tracking_confidence=0.1) as holistic:
        results = holistic.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        # image_height, image_width, _ = image.shape

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


for i in range(24, 25):

    paths = "D:\\Software\\PyCharm\\AI\\MediaPipe\\dataset\\" + str(i).rjust(3, '0')
    csv_path = "D:\\Software\\PyCharm\\AI\\data\\" + str(i).rjust(3, '0')

    if not os.path.exists(csv_path):
        os.mkdir(csv_path)

    for dirlist in os.listdir(paths):
        for root, directories, filenames in os.walk(os.path.join(paths, dirlist)):
            print("Inside Folder", dirlist, "Consist :", len(filenames), "Imageset")
            csv_path_final = csv_path + "\\" + dirlist + ".csv"

            if os.path.exists(csv_path_final):
                print("CSV File does exist")
                # os.remove(csv_path_final)
                continue
            else:
                print("The CSV file does not exist", csv_path_final, ",Going Create after Extraction")

            for filename in filenames:
                if filename.endswith(".jpg") or filename.endswith(".JPG"):
                    # print(os.path.join(root, filename), True)
                    [left_shoulderX, left_shoulderY,
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
                     right_pinky_TipX, right_pinky_TipY] = extract_feature(os.path.join(root, filename))

                    if (left_shoulderX != np.nan) and (left_wristX != np.nan):
                        toCSV(csv_path_final, str(i).rjust(3, '0'),
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

                    else:
                        print(os.path.join(root, filename), "Pose does not have landmarks")

            data = pd.read_csv(csv_path_final, header=0, sep=',')
            if not data.iloc[0:4, :].isnull().any().any():
                data = data.interpolate(method='cubic', limit_direction='both')
                data.to_csv(csv_path_final)

    print("===================Feature Extraction for TRAINING is Completed===================")
