# Import the much needed stuff for training
import csv
import os
import torch.nn.modules.activation
import cv2 as cv
import mediapipe as mp
import numpy as np

np.set_printoptions(precision=3, suppress=True)  # 禁用科学计数法，设置小数保留后三位


# Function to Extract Feature from images or Frame
def extract_feature(input_image):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    image = cv.imread(input_image)
    with mp_pose.Pose(static_image_mode=True, model_complexity=0,
                      enable_segmentation=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv.flip(cv.cvtColor(image, cv.COLOR_BGR2RGB), 1))
        image_height, image_width, _ = image.shape

        if results.pose_landmarks:

            left_shoulderX = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
            left_shoulderY = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
            left_shoulderZ = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z

            right_shoulderX = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
            right_shoulderY = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
            right_shoulderZ = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z

            left_elbowX = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width
            left_elbowY = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height
            left_elbowZ = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z

            right_elbowX = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width
            right_elbowY = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height
            right_elbowZ = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z

            # Crotch 23-24
            left_crotchX = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width
            left_crotchY = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height
            left_crotchZ = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z

            right_crotchX = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width
            right_crotchY = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height
            right_crotchZ = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z

            features_pose = [left_shoulderX, left_shoulderY, left_shoulderZ,
                             right_shoulderX, right_shoulderY, right_shoulderZ,
                             left_elbowX, left_elbowY, left_elbowZ,
                             right_elbowX, right_elbowY, right_elbowZ,
                             left_crotchX, left_crotchY, left_crotchZ,
                             right_crotchX, right_crotchY, right_crotchZ]

        else:
            # Shoulder 11-12
            left_shoulderX = 0
            left_shoulderY = 0
            left_shoulderZ = 0

            right_shoulderX = 0
            right_shoulderY = 0
            right_shoulderZ = 0

            # Elbow 13-14
            left_elbowX = 0
            left_elbowY = 0
            left_elbowZ = 0

            right_elbowX = 0
            right_elbowY = 0
            right_elbowZ = 0

            # Crotch 23-24
            left_crotchX = 0
            left_crotchY = 0
            left_crotchZ = 0

            right_crotchX = 0
            right_crotchY = 0
            right_crotchZ = 0

            features_pose = [left_shoulderX, left_shoulderY, left_shoulderZ,
                             right_shoulderX, right_shoulderY, right_shoulderZ,
                             left_elbowX, left_elbowY, left_elbowZ,
                             right_elbowX, right_elbowY, right_elbowZ,
                             left_crotchX, left_crotchY, left_crotchZ,
                             right_crotchX, right_crotchY, right_crotchZ]

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1) as hands:
        results2 = hands.process(cv.flip(cv.cvtColor(image, cv.COLOR_BGR2RGB), 1))
        image_height, image_width, _ = image.shape

        if results2.multi_hand_landmarks:
            # Wrist Hand /  Pergelangan Tangan
            for hand_landmarks in results2.multi_hand_landmarks:
                wristX = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
                wristY = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
                wristZ = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

                # Thumb Finger / Ibu Jari
                thumb_CmcX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width
                thumb_CmcY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height
                thumb_CmcZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z

                thumb_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width
                thumb_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height
                thumb_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z

                thumb_IpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width
                thumb_IpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height
                thumb_IpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z

                thumb_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
                thumb_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
                thumb_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

                # Index Finger / Jari Telunjuk
                index_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width
                index_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
                index_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

                index_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width
                index_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height
                index_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z

                index_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width
                index_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height
                index_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z

                index_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                index_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                index_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

                # Middle Finger / Jari Tengah
                middle_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
                middle_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height
                middle_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z

                middle_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width
                middle_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height
                middle_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z

                middle_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width
                middle_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height
                middle_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z

                middle_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
                middle_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height
                middle_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z

                # Ring Finger / Jari Cincin
                ring_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width
                ring_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height
                ring_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z

                ring_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width
                ring_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height
                ring_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z

                ring_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width
                ring_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height
                ring_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z

                ring_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width
                ring_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height
                ring_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z

                # Pinky Finger / Jari Kelingking
                pinky_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width
                pinky_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height
                pinky_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z

                pinky_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width
                pinky_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height
                pinky_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z

                pinky_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width
                pinky_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height
                pinky_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z

                pinky_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width
                pinky_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height
                pinky_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z

            features_hands = [wristX, wristY, wristZ,
                                  thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                                  thumb_McpX, thumb_McpY, thumb_McpZ,
                                  thumb_IpX, thumb_IpY, thumb_IpZ,
                                  thumb_TipX, thumb_TipY, thumb_TipZ,
                                  index_McpX, index_McpY, index_McpZ,
                                  index_PipX, index_PipY, index_PipZ,
                                  index_DipX, index_DipY, index_DipZ,
                                  index_TipX, index_TipY, index_TipZ,
                                  middle_McpX, middle_McpY, middle_McpZ,
                                  middle_PipX, middle_PipY, middle_PipZ,
                                  middle_DipX, middle_DipY, middle_DipZ,
                                  middle_TipX, middle_TipY, middle_TipZ,
                                  ring_McpX, ring_McpY, ring_McpZ,
                                  ring_PipX, ring_PipY, ring_PipZ,
                                  ring_DipX, ring_DipY, ring_DipZ,
                                  ring_TipX, ring_TipY, ring_TipZ,
                                  pinky_McpX, pinky_McpY, pinky_McpZ,
                                  pinky_PipX, pinky_PipY, pinky_PipZ,
                                  pinky_DipX, pinky_DipY, pinky_DipZ,
                                  pinky_TipX, pinky_TipY, pinky_TipZ]

        else:
            # Here we will set whole landmarks into zero as no handpose detected
            # in a picture wanted to extract.

            # Wrist Hand
            wristX = 0
            wristY = 0
            wristZ = 0

            # Thumb Finger
            thumb_CmcX = 0
            thumb_CmcY = 0
            thumb_CmcZ = 0

            thumb_McpX = 0
            thumb_McpY = 0
            thumb_McpZ = 0

            thumb_IpX = 0
            thumb_IpY = 0
            thumb_IpZ = 0

            thumb_TipX = 0
            thumb_TipY = 0
            thumb_TipZ = 0

            # Index Finger
            index_McpX = 0
            index_McpY = 0
            index_McpZ = 0

            index_PipX = 0
            index_PipY = 0
            index_PipZ = 0

            index_DipX = 0
            index_DipY = 0
            index_DipZ = 0

            index_TipX = 0
            index_TipY = 0
            index_TipZ = 0

            # Middle Finger
            middle_McpX = 0
            middle_McpY = 0
            middle_McpZ = 0

            middle_PipX = 0
            middle_PipY = 0
            middle_PipZ = 0

            middle_DipX = 0
            middle_DipY = 0
            middle_DipZ = 0

            middle_TipX = 0
            middle_TipY = 0
            middle_TipZ = 0

            # Ring Finger
            ring_McpX = 0
            ring_McpY = 0
            ring_McpZ = 0

            ring_PipX = 0
            ring_PipY = 0
            ring_PipZ = 0

            ring_DipX = 0
            ring_DipY = 0
            ring_DipZ = 0

            ring_TipX = 0
            ring_TipY = 0
            ring_TipZ = 0

            # Pinky Finger
            pinky_McpX = 0
            pinky_McpY = 0
            pinky_McpZ = 0

            pinky_PipX = 0
            pinky_PipY = 0
            pinky_PipZ = 0

            pinky_DipX = 0
            pinky_DipY = 0
            pinky_DipZ = 0

            pinky_TipX = 0
            pinky_TipY = 0
            pinky_TipZ = 0

            # Return Whole Landmark and Image
            features_hands = [wristX, wristY, wristZ,
                              thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                              thumb_McpX, thumb_McpY, thumb_McpZ,
                              thumb_IpX, thumb_IpY, thumb_IpZ,
                              thumb_TipX, thumb_TipY, thumb_TipZ,
                              index_McpX, index_McpY, index_McpZ,
                              index_PipX, index_PipY, index_PipZ,
                              index_DipX, index_DipY, index_DipZ,
                              index_TipX, index_TipY, index_TipZ,
                              middle_McpX, middle_McpY, middle_McpZ,
                              middle_PipX, middle_PipY, middle_PipZ,
                              middle_DipX, middle_DipY, middle_DipZ,
                              middle_TipX, middle_TipY, middle_TipZ,
                              ring_McpX, ring_McpY, ring_McpZ,
                              ring_PipX, ring_PipY, ring_PipZ,
                              ring_DipX, ring_DipY, ring_DipZ,
                              ring_TipX, ring_TipY, ring_TipZ,
                              pinky_McpX, pinky_McpY, pinky_McpZ,
                              pinky_PipX, pinky_PipY, pinky_PipZ,
                              pinky_DipX, pinky_DipY, pinky_DipZ,
                              pinky_TipX, pinky_TipY, pinky_TipZ]
        features = features_pose + features_hands
        return features


# Function to create CSV file or add dataset to the existed CSV file
def toCSV(filecsv, class_type,
          left_shoulderX, left_shoulderY, left_shoulderZ,
          right_shoulderX, right_shoulderY, right_shoulderZ,
          left_elbowX, left_elbowY, left_elbowZ,
          right_elbowX, right_elbowY, right_elbowZ,
          left_crotchX, left_crotchY, left_crotchZ,
          right_crotchX, right_crotchY, right_crotchZ,
          wristX, wristY, wristZ,
          thumb_CmcX, thumb_CmcY, thumb_CmcZ,
          thumb_McpX, thumb_McpY, thumb_McpZ,
          thumb_IpX, thumb_IpY, thumb_IpZ,
          thumb_TipX, thumb_TipY, thumb_TipZ,
          index_McpX, index_McpY, index_McpZ,
          index_PipX, index_PipY, index_PipZ,
          index_DipX, index_DipY, index_DipZ,
          index_TipX, index_TipY, index_TipZ,
          middle_McpX, middle_McpY, middle_McpZ,
          middle_PipX, middle_PipY, middle_PipZ,
          middle_DipX, middle_DipY, middle_DipZ,
          middle_TipX, middle_TipY, middle_TipZ,
          ring_McpX, ring_McpY, ring_McpZ,
          ring_PipX, ring_PipY, ring_PipZ,
          ring_DipX, ring_DipY, ring_DipZ,
          ring_TipX, ring_TipY, ring_TipZ,
          pinky_McpX, pinky_McpY, pinky_McpZ,
          pinky_PipX, pinky_PipY, pinky_PipZ,
          pinky_DipX, pinky_DipY, pinky_DipZ,
          pinky_TipX, pinky_TipY, pinky_TipZ):
    if os.path.isfile(filecsv):
        # print ("File exist thus shall write append to the file")
        with open(filecsv, 'a+', newline='') as file:
            # Create a writer object from csv module
            writer = csv.writer(file)
            writer.writerow([class_type,
                             left_shoulderX, left_shoulderY, left_shoulderZ,
                             right_shoulderX, right_shoulderY, right_shoulderZ,
                             left_elbowX, left_elbowY, left_elbowZ,
                             right_elbowX, right_elbowY, right_elbowZ,
                             left_crotchX, left_crotchY, left_crotchZ,
                             right_crotchX, right_crotchY, right_crotchZ,
                             wristX, wristY, wristZ,
                             thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                             thumb_McpX, thumb_McpY, thumb_McpZ,
                             thumb_IpX, thumb_IpY, thumb_IpZ,
                             thumb_TipX, thumb_TipY, thumb_TipZ,
                             index_McpX, index_McpY, index_McpZ,
                             index_PipX, index_PipY, index_PipZ,
                             index_DipX, index_DipY, index_DipZ,
                             index_TipX, index_TipY, index_TipZ,
                             middle_McpX, middle_McpY, middle_McpZ,
                             middle_PipX, middle_PipY, middle_PipZ,
                             middle_DipX, middle_DipY, middle_DipZ,
                             middle_TipX, middle_TipY, middle_TipZ,
                             ring_McpX, ring_McpY, ring_McpZ,
                             ring_PipX, ring_PipY, ring_PipZ,
                             ring_DipX, ring_DipY, ring_DipZ,
                             ring_TipX, ring_TipY, ring_TipZ,
                             pinky_McpX, pinky_McpY, pinky_McpZ,
                             pinky_PipX, pinky_PipY, pinky_PipZ,
                             pinky_DipX, pinky_DipY, pinky_DipZ,
                             pinky_TipX, pinky_TipY, pinky_TipZ])
    else:
        # print ("File not exist thus shall create new file as", filecsv)
        with open(filecsv, 'w', newline='') as file:
            # Create a writer object from csv module
            writer = csv.writer(file)
            writer.writerow(["class_type",
                             'left_shoulderX', 'left_shoulderY', 'left_shoulderZ',
                             'right_shoulderX', 'right_shoulderY', 'right_shoulderZ',
                             'left_elbowX', 'left_elbowY', 'left_elbowZ',
                             'right_elbowX', 'right_elbowY', 'right_elbowZ',
                             'left_crotchX', 'left_crotchY', 'left_crotchZ',
                             'right_crotchX', 'right_crotchY', 'right_crotchZ',
                             "wristX", "wristY", "wristZ",
                             "thumb_CmcX", "thumb_CmcY", "thumb_CmcZ",
                             "thumb_McpX", "thumb_McpY", "thumb_McpZ",
                             "thumb_IpX", "thumb_IpY", "thumb_IpZ",
                             "thumb_TipX", "thumb_TipY", "thumb_TipZ",
                             "index_McpX", "index_McpY", "index_McpZ",
                             "index_PipX", "index_PipY", "index_PipZ",
                             "index_DipX", "index_DipY", "index_DipZ",
                             "index_TipX", "index_TipY", "index_TipZ",
                             "middle_McpX", "middle_McpY", "middle_McpZ",
                             "middle_PipX", "middle_PipY", "middle_PipZ",
                             "middle_DipX", "middle_DipY", "middle_DipZ",
                             "middle_TipX", "middle_TipY", "middle_TipZ",
                             "ring_McpX", "ring_McpY", "ring_McpZ",
                             "ring_PipX", "ring_PipY", "ring_PipZ",
                             "ring_DipX", "ring_DipY", "ring_DipZ",
                             "ring_TipX", "ring_TipY", "ring_TipZ",
                             "pinky_McpX", "pinky_McpY", "pinky_McpZ",
                             "pinky_PipX", "pinky_PipY", "pinky_PipZ",
                             "pinky_DipX", "pinky_DipY", "pinky_DipZ",
                             "pinky_TipX", "pinky_TipY", "pinky_TipZ"])
            writer.writerow([class_type,
                             left_shoulderX, left_shoulderY, left_shoulderZ,
                             right_shoulderX, right_shoulderY, right_shoulderZ,
                             left_elbowX, left_elbowY, left_elbowZ,
                             right_elbowX, right_elbowY, right_elbowZ,
                             left_crotchX, left_crotchY, left_crotchZ,
                             right_crotchX, right_crotchY, right_crotchZ,
                             wristX, wristY, wristZ,
                             thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                             thumb_McpX, thumb_McpY, thumb_McpZ,
                             thumb_IpX, thumb_IpY, thumb_IpZ,
                             thumb_TipX, thumb_TipY, thumb_TipZ,
                             index_McpX, index_McpY, index_McpZ,
                             index_PipX, index_PipY, index_PipZ,
                             index_DipX, index_DipY, index_DipZ,
                             index_TipX, index_TipY, index_TipZ,
                             middle_McpX, middle_McpY, middle_McpZ,
                             middle_PipX, middle_PipY, middle_PipZ,
                             middle_DipX, middle_DipY, middle_DipZ,
                             middle_TipX, middle_TipY, middle_TipZ,
                             ring_McpX, ring_McpY, ring_McpZ,
                             ring_PipX, ring_PipY, ring_PipZ,
                             ring_DipX, ring_DipY, ring_DipZ,
                             ring_TipX, ring_TipY, ring_TipZ,
                             pinky_McpX, pinky_McpY, pinky_McpZ,
                             pinky_PipX, pinky_PipY, pinky_PipZ,
                             pinky_DipX, pinky_DipY, pinky_DipZ,
                             pinky_TipX, pinky_TipY, pinky_TipZ])


for i in range(41, 50):

    paths = "D:\\Software\\PyCharm\\AI\\MediaPipe\\dataset\\" + str(i).rjust(3, '0')
    csv_path = "D:\\Software\\PyCharm\\AI\\MediaPipe\\data\\" + str(i).rjust(3, '0')
    os.mkdir(csv_path)

    for dirlist in os.listdir(paths):
        for root, directories, filenames in os.walk(os.path.join(paths, dirlist)):
            print("Inside Folder", dirlist, "Consist :", len(filenames), "Imageset")
            csv_path_final = csv_path + "\\" + dirlist + ".csv"

            if os.path.exists(csv_path_final):
                print("CSV File does exist, going delete before start extraction and replace it with new")
                os.remove(csv_path_final)
            else:
                print("The CSV file does not exist", csv_path_final, ",Going Create after Extraction")

            for filename in filenames:
                if filename.endswith(".jpg") or filename.endswith(".JPG"):
                    # print(os.path.join(root, filename), True)
                    [left_shoulderX, left_shoulderY, left_shoulderZ,
                     right_shoulderX, right_shoulderY, right_shoulderZ,
                     left_elbowX, left_elbowY, left_elbowZ,
                     right_elbowX, right_elbowY, right_elbowZ,
                     left_crotchX, left_crotchY, left_crotchZ,
                     right_crotchX, right_crotchY, right_crotchZ,
                     wristX, wristY, wristZ,
                     thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                     thumb_McpX, thumb_McpY, thumb_McpZ,
                     thumb_IpX, thumb_IpY, thumb_IpZ,
                     thumb_TipX, thumb_TipY, thumb_TipZ,
                     index_McpX, index_McpY, index_McpZ,
                     index_PipX, index_PipY, index_PipZ,
                     index_DipX, index_DipY, index_DipZ,
                     index_TipX, index_TipY, index_TipZ,
                     middle_McpX, middle_McpY, middle_McpZ,
                     middle_PipX, middle_PipY, middle_PipZ,
                     middle_DipX, middle_DipY, middle_DipZ,
                     middle_TipX, middle_TipY, middle_TipZ,
                     ring_McpX, ring_McpY, ring_McpZ,
                     ring_PipX, ring_PipY, ring_PipZ,
                     ring_DipX, ring_DipY, ring_DipZ,
                     ring_TipX, ring_TipY, ring_TipZ,
                     pinky_McpX, pinky_McpY, pinky_McpZ,
                     pinky_PipX, pinky_PipY, pinky_PipZ,
                     pinky_DipX, pinky_DipY, pinky_DipZ,
                     pinky_TipX, pinky_TipY, pinky_TipZ] = extract_feature(os.path.join(root, filename))

                    if (not left_shoulderX == 0) and (not left_shoulderY == 0):
                        toCSV(csv_path_final, str(i).rjust(3, '0'),
                              left_shoulderX, left_shoulderY, left_shoulderZ,
                              right_shoulderX, right_shoulderY, right_shoulderZ,
                              left_elbowX, left_elbowY, left_elbowZ,
                              right_elbowX, right_elbowY, right_elbowZ,
                              left_crotchX, left_crotchY, left_crotchZ,
                              right_crotchX, right_crotchY, right_crotchZ,
                              wristX, wristY, wristZ,
                              thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                              thumb_McpX, thumb_McpY, thumb_McpZ,
                              thumb_IpX, thumb_IpY, thumb_IpZ,
                              thumb_TipX, thumb_TipY, thumb_TipZ,
                              index_McpX, index_McpY, index_McpZ,
                              index_PipX, index_PipY, index_PipZ,
                              index_DipX, index_DipY, index_DipZ,
                              index_TipX, index_TipY, index_TipZ,
                              middle_McpX, middle_McpY, middle_McpZ,
                              middle_PipX, middle_PipY, middle_PipZ,
                              middle_DipX, middle_DipY, middle_DipZ,
                              middle_TipX, middle_TipY, middle_TipZ,
                              ring_McpX, ring_McpY, ring_McpZ,
                              ring_PipX, ring_PipY, ring_PipZ,
                              ring_DipX, ring_DipY, ring_DipZ,
                              ring_TipX, ring_TipY, ring_TipZ,
                              pinky_McpX, pinky_McpY, pinky_McpZ,
                              pinky_PipX, pinky_PipY, pinky_PipZ,
                              pinky_DipX, pinky_DipY, pinky_DipZ,
                              pinky_TipX, pinky_TipY, pinky_TipZ)

                    else:
                        print(os.path.join(root, filename), "Pose does not have landmarks")

    print("===================Feature Extraction for TRAINING is Completed===================")
