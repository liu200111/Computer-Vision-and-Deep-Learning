import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFrame, QSpacerItem, QSizePolicy, QLineEdit, QLabel, QMessageBox
from PyQt5.QtGui import QFont

import os
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

# UI
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1000, 870)
        Form.setWindowTitle("CvDl_Hw1_P66134145")
        
        self.centralwidget = QtWidgets.QWidget(Form)
        #self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(30, 30, 250, 300))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(15, 30, 220, 250))
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.button_1_1 = QtWidgets.QPushButton("Load Folder", self.verticalLayoutWidget)
        self.filePathLabel_1 = QLabel("", self.verticalLayoutWidget)
        self.button_1_2 = QtWidgets.QPushButton("Load Image_L", self.verticalLayoutWidget)
        self.filePathLabel_2 = QLabel("", self.verticalLayoutWidget)
        self.button_1_3 = QtWidgets.QPushButton("Load Image_R", self.verticalLayoutWidget)
        self.filePathLabel_3 = QLabel("", self.verticalLayoutWidget)
        self.verticalLayout.addWidget(self.button_1_1)
        self.verticalLayout.addWidget(self.filePathLabel_1)
        self.verticalLayout.addWidget(self.button_1_2)
        self.verticalLayout.addWidget(self.filePathLabel_2)
        self.verticalLayout.addWidget(self.button_1_3)
        self.verticalLayout.addWidget(self.filePathLabel_3)
        self.label = QtWidgets.QLabel("Load Image", self.frame)
        #self.label.setEnabled(False)
        self.label.setGeometry(QtCore.QRect(10, 5, 100, 20))
        font = QFont()      #Bold
        font.setBold(True)
        self.label.setFont(font)
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        #self.label.setSizePolicy(sizePolicy)
        
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(310, 30, 300, 390))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.frame_2)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(20, 30, 260, 350))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.button_2_1 = QtWidgets.QPushButton("1.1 Find Corners", self.verticalLayoutWidget_2)
        self.button_2_2 = QtWidgets.QPushButton("1.2 Find Intrinsic", self.verticalLayoutWidget_2)
        
        self.frame_2_1 = QtWidgets.QFrame(self.verticalLayoutWidget_2)
        self.frame_2_1.setGeometry(QtCore.QRect(5, 115, 246, 120))
        self.frame_2_1.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.button_2_3 = QtWidgets.QPushButton("1.3 Find Extrinsic", self.frame_2_1)
        self.button_2_3.setGeometry(QtCore.QRect(50, 70, 150, 40))
        self.comboBox = QtWidgets.QComboBox(self.frame_2_1)
        self.comboBox.setGeometry(QtCore.QRect(50, 20, 150, 40))
        for i in range(1, 16):  # add item from 1 to 15
            self.comboBox.addItem(str(i))
        
        self.button_2_4 = QtWidgets.QPushButton("1.4 Find Distortion", self.verticalLayoutWidget_2)
        self.button_2_5 = QtWidgets.QPushButton("1.5 Show Result", self.verticalLayoutWidget_2)
        self.verticalLayout_2.addWidget(self.button_2_1)
        self.verticalLayout_2.addWidget(self.button_2_2)
        spacer = QSpacerItem(0, 140, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacer)
        self.verticalLayout_2.addWidget(self.button_2_4)
        self.verticalLayout_2.addWidget(self.button_2_5)
        self.label_2 = QtWidgets.QLabel("1.Calibration", self.frame_2)
        self.label_2.setGeometry(QtCore.QRect(10, 5, 150, 20))
        self.label_2.setFont(font)
        
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(650, 30, 300, 390))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.frame_3)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(20, 30, 260, 350))
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label_3 = QtWidgets.QLabel("2. Augmented Reality", self.frame_3)
        self.label_3.setGeometry(QtCore.QRect(10, 5, 150, 20))
        self.label_3.setFont(font)
        self.lineEdit = QLineEdit(self.verticalLayoutWidget_3)
        self.button_3_1 = QtWidgets.QPushButton("2.1 Show Words on Board", self.verticalLayoutWidget_3)
        self.button_3_2 = QtWidgets.QPushButton("2.2 Show Words Vertically", self.verticalLayoutWidget_3)
        self.verticalLayout_3.addWidget(self.lineEdit)
        self.verticalLayout_3.addWidget(self.button_3_1)
        self.verticalLayout_3.addWidget(self.button_3_2)
        
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(310, 450, 300, 390))
        self.frame_4.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.frame_4)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(20, 30, 260, 350))
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.label_4 = QtWidgets.QLabel("3. Stereo Disparity Map", self.frame_4)
        self.label_4.setGeometry(QtCore.QRect(10, 5, 150, 20))
        self.label_4.setFont(font)
        self.button_4 = QtWidgets.QPushButton("3.1 Stereo Disparity Map", self.verticalLayoutWidget_4)
        self.verticalLayout_4.addWidget(self.button_4)
        
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(650, 450, 300, 390))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.frame_5)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(20, 30, 260, 350))
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label_5 = QtWidgets.QLabel("4. SIFT", self.frame_5)
        self.label_5.setGeometry(QtCore.QRect(10, 5, 150, 20))
        self.label_5.setFont(font)
        self.button_5_1 = QtWidgets.QPushButton("4.1 SIFT Keypoints", self.verticalLayoutWidget_5)
        self.button_5_2 = QtWidgets.QPushButton("4.2 Matched Keypoints", self.verticalLayoutWidget_5)
        self.verticalLayout_5.addWidget(self.button_5_1)
        self.verticalLayout_5.addWidget(self.button_5_2)
        
    
# methods and operations
class Main_Frame(QFrame, Ui_Form):
    def __init__(self):
        super(Main_Frame, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        # Load data
        self.ui.button_1_1.clicked.connect(self.load_folder)
        self.ui.button_1_2.clicked.connect(self.load_image_L)
        self.ui.button_1_3.clicked.connect(self.load_image_R)
        
        # Q1 Camera Calibration
        self.ui.button_2_1.clicked.connect(self.find_corners)
        self.ui.button_2_2.clicked.connect(self.find_intrinsic)
        self.ui.button_2_3.clicked.connect(self.find_extrinsic)
        self.ui.button_2_4.clicked.connect(self.find_distortion)
        self.ui.button_2_5.clicked.connect(self.show_result)
        
        # Q2 Augmented Reality 
        self.ui.button_3_1.clicked.connect(self.show_words_on_board)
        self.ui.button_3_2.clicked.connect(self.show_words_vertically)
        
        # Q3 Stereo Disparity Map
        self.ui.button_4.clicked.connect(self.stereo_disparity_map)
        
        # Q4 SIFT
        self.ui.button_5_1.clicked.connect(self.SIFT_keypoints)
        self.ui.button_5_2.clicked.connect(self.matched_keypoints)
        
        # set initial value
        self.corners_found = False
        self.SIFT_match = False
        
    def load_folder(self):
        self.folderPath = QtWidgets.QFileDialog.getExistingDirectory()
        self.ui.filePathLabel_1.setText("Folder name: " + f"{os.path.basename(self.folderPath)}")  # show the filepath
        self.ui.filePathLabel_1.setStyleSheet("font-size: 14px;")
        self.ui.filePathLabel_1.setWordWrap(True)   #automatically wrap lines
    
    def load_image_L(self):
        self.file_path_L, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "select image_L", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        self.ui.filePathLabel_2.setText("Image name: " + f"{os.path.basename(self.file_path_L)}")  # show the filepath
        self.ui.filePathLabel_2.setStyleSheet("font-size: 14px;")
        
    def load_image_R(self):
        self.file_path_R, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "select image_R", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        self.ui.filePathLabel_3.setText("Image name: " + f"{os.path.basename(self.file_path_R)}")  # show the filepath
        self.ui.filePathLabel_3.setStyleSheet("font-size: 14px;")
    
    def find_corners(self):
        pattern_size = (11, 8)    # width and high of corner in chessboard
        winSize = (5, 5)    # the range of the search area near the corner point
        zeroZone = (-1, -1)    # window size prevent from focusing on edge of image
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
 
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objectPoints = []  # 3D points in real world space
        imagePoints = []   # 2D points in image plane
 
        self.images = glob.glob(os.path.join(self.folderPath, "*.bmp"))
        for image_path in self.images:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, pattern_size)    #detect corners
            if ret:
                corners = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)  # improve corner accuracy
                objectPoints.append(objp)
                imagePoints.append(corners)
                cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                cv2.imshow("Chessboard Corners", cv2.resize(img,(512,512)))
                cv2.waitKey(100)
        #cv2.destroyAllWindows()
        
        # calibration
        ret, self.ins, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera (objectPoints, imagePoints, gray.shape[::-1], None, None)
        self.corners_found = True
        
    def find_intrinsic(self):
        #print('Intrinsic Matrix: ')
        #print(self.ins)
        matrix_str = np.array2string(self.ins, precision=3, separator=', ')
        msg = QMessageBox()
        msg.setWindowTitle("Intrinsic Matrix")
        msg.setText(matrix_str)
        msg.exec_()
    
    def find_extrinsic(self):
        index = int(self.ui.comboBox.currentText())
        rmtx, jacobin = cv2.Rodrigues(self.rvecs[index - 1])
        extrinsic = np.hstack((rmtx, self.tvecs[index - 1]))
        matrix_str = np.array2string(extrinsic, precision=3, separator=', ')
        msg = QMessageBox()
        msg.setWindowTitle("Extrinsic Matrix")
        msg.setText(matrix_str)
        msg.exec_()
        
    def find_distortion(self):
        matrix_str = np.array2string(self.dist, precision=3, separator=', ')
        msg = QMessageBox()
        msg.setWindowTitle("Distortion Matrix")
        msg.setText(matrix_str)
        msg.exec_()
        
    def show_result(self):
        images = glob.glob(os.path.join(self.folderPath, "*.bmp"))
        for image_path in images:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result_img = cv2.undistort(gray, self.ins, self.dist)
            plt.figure(figsize = (10, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(gray, cmap = 'gray')
            plt.subplot(1, 2, 2)
            plt.title("Undistorted Image")
            plt.imshow(result_img, cmap = 'gray')
            plt.show()
            #cv2.imshow(os.path.basename(image_path), cv2.resize(result_img,(512,512)))
            #cv2.waitKey(500)
            
    def show_words_on_board(self):
        if self.corners_found:
            None
        else:
            self.find_corners()     # camera calibration
            cv2.destroyAllWindows()
        
        self.word = str(self.ui.lineEdit.text())    # get text
        charOrigin = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]]).astype(np.float32)    # char bottom
        nodes = np.empty((0, 2, 3), dtype=np.float32)

        fs = cv2.FileStorage('Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_onboard.txt', cv2.FILE_STORAGE_READ)    # read the alphabet_db
        for i, char in enumerate(self.word):
            charPoints = fs.getNode(char).mat()    # 2D
            charPoints = charPoints + charOrigin[i]    # 2D corrected
            nodes = np.append(nodes, charPoints, axis=0)
        
        for i, image_path in enumerate(self.images):
            img = cv2.imread(image_path)
            img_copy = img.copy()    # no modifications to the original image
            for objpoints_line in nodes:
                imgpoints_line, jmtx = cv2.projectPoints(
                    objpoints_line, self.rvecs[i], self.tvecs[i], self.ins, self.dist)    # project to board
                imgpoints_line = np.squeeze(imgpoints_line).astype(int)
                cv2.line(img_copy, tuple(imgpoints_line[0]), tuple(imgpoints_line[1]), (0, 0, 255), 10)
            cv2.imshow('Words are displayed on board',cv2.resize(img_copy,(512,512)))
            #cv2.waitKey(2000)
            cv2.waitKey(1000)
            
    def show_words_vertically(self):
        if self.corners_found:
            None
        else:
            self.find_corners()     # camera calibration
            cv2.destroyAllWindows()
        
        self.word = str(self.ui.lineEdit.text())    # get text
        charOrigin = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]]).astype(np.float32)    # char bottom
        nodes = np.empty((0, 2, 3), dtype=np.float32)

        fs = cv2.FileStorage('Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_vertical.txt', cv2.FILE_STORAGE_READ)    # read the alphabet_db
        for i, char in enumerate(self.word):
            charPoints = fs.getNode(char).mat()    # 2D
            charPoints = charPoints + charOrigin[i]    # 2D corrected
            nodes = np.append(nodes, charPoints, axis=0)
        
        for i, image_path in enumerate(self.images):
            img = cv2.imread(image_path)
            img_copy = img.copy()    # no modifications to the original image
            for objpoints_line in nodes:
                imgpoints_line, jmtx = cv2.projectPoints(
                    objpoints_line, self.rvecs[i], self.tvecs[i], self.ins, self.dist)    # project to board
                imgpoints_line = np.squeeze(imgpoints_line).astype(int)
                cv2.line(img_copy, tuple(imgpoints_line[0]), tuple(imgpoints_line[1]), (0, 0, 255), 10)
            cv2.imshow('Words are displayed vertically',cv2.resize(img_copy,(512,512)))
            cv2.waitKey(1000)

    def stereo_disparity_map(self):
        '''
        1. numDisparities:
            The larger the value, the wider the search range.
        2. blockSize:
            The larger the block, the smoother the disparity map, but the accuracy decreases;
            the smaller the block, the more details, but the possibility of false comparison increases.
        '''
        numDisparities = 160  # confirm whether it is a multiple of 16 and suitable for the scene
        blockSize = 25  # confirm whether it is between [5, 51] and is an odd number
        stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
        
        imgL = cv2.imread(self.file_path_L, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(self.file_path_R, cv2.IMREAD_GRAYSCALE)
        disparity = stereo.compute(imgL, imgR)

        # normalize the disparity map to [0, 255]
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_normalized = np.uint8(disparity_normalized)

        cv2.imshow('Left Image', cv2.resize(imgL, (640, 360)))
        cv2.imshow('Right Image', cv2.resize(imgR, (640, 360)))
        cv2.imshow('Disparity Map', cv2.resize(disparity_normalized, (640, 360)))
        cv2.waitKey(0)
    
    def SIFT_keypoints(self):
        img_1 = cv2.imread(self.file_path_L, cv2.IMREAD_GRAYSCALE)
        #img_2 = cv2.imread(self.file_path_R, cv2.IMREAD_GRAYSCALE)
        self.img_1_copy = img_1.copy()
        #self.img_2_copy = img_2.copy()
        
        sift = cv2.SIFT_create()      # Create a SIFT detector
        keypoints, descriptors = sift.detectAndCompute(img_1, None)
        img_1_kp = cv2.drawKeypoints(self.img_1_copy, keypoints, None, color=(0,255,0))
        
        cv2.imshow('Left image with keypoints', cv2.resize(img_1_kp, (512,512)))
        cv2.waitKey(0)
        
    def matched_keypoints(self):
        img_1 = cv2.imread(self.file_path_L, cv2.IMREAD_GRAYSCALE)
        img_2 = cv2.imread(self.file_path_R, cv2.IMREAD_GRAYSCALE)
        self.img_1_copy = img_1.copy()
        self.img_2_copy = img_2.copy()
        sift = cv2.SIFT_create()      # Create a SIFT detector
        keypoints_1, descriptors_1 = sift.detectAndCompute(img_1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img_2, None)
        
        # find match keypoints of two images
        '''
        k=2 -> means that each feature point will find the two closest matching points
        '''
        matches = cv2.BFMatcher().knnMatch(descriptors_1, descriptors_2, k=2)
        
        # Ratio Test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        img_matches = cv2.drawMatchesKnn(self.img_1_copy, keypoints_1, self.img_2_copy, keypoints_2,
                           good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow('Matching result', cv2.resize(img_matches, (1024,512)))
        cv2.waitKey(0)
        self.SIFT_match = True
        

# execution program
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainFrame = Main_Frame()
    mainFrame.show()
    sys.exit(app.exec_())