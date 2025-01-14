from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QLabel, QVBoxLayout
from PyQt5.QtGui import QFont, QPixmap

# UI
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(900, 450)
        Form.setWindowTitle("CvDl_Hw1_P66134145")
        Form.setFont(QFont("Times New Roman", 8))
        font = QFont()      #Bold
        font.setBold(True)
        
        self.centralwidget = QtWidgets.QWidget(Form)
        
        # Q1
        self.frame_1 = QtWidgets.QFrame(self.centralwidget)
        self.frame_1.setGeometry(QtCore.QRect(30, 25, 250, 400))
        self.frame_1.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayoutWidget_1 = QtWidgets.QWidget(self.frame_1)
        self.verticalLayoutWidget_1.setGeometry(QtCore.QRect(15, 30, 220, 350))
        #self.verticalLayoutWidget.setStyleSheet("QPushButton { font-size: 12pt; }")
        self.verticalLayout_1 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_1)
        self.verticalLayout_1.setContentsMargins(0, 0, 0, 0)
        self.button_1_1 = QtWidgets.QPushButton("Load Image", self.verticalLayoutWidget_1)
        self.button_1_2 = QtWidgets.QPushButton("1. Show Augmented Images", self.verticalLayoutWidget_1)
        self.button_1_3 = QtWidgets.QPushButton("2. Show Model Structure", self.verticalLayoutWidget_1)
        self.button_1_4 = QtWidgets.QPushButton("3. Show Accuracy and Loss", self.verticalLayoutWidget_1)
        self.button_1_5 = QtWidgets.QPushButton("4. Inference", self.verticalLayoutWidget_1)
        self.verticalLayout_1.addWidget(self.button_1_1)
        self.verticalLayout_1.addWidget(self.button_1_2)
        self.verticalLayout_1.addWidget(self.button_1_3)
        self.verticalLayout_1.addWidget(self.button_1_4)
        self.verticalLayout_1.addWidget(self.button_1_5)
        self.button_1_1.setFixedSize(220, 50)
        self.button_1_2.setFixedSize(220, 50)
        self.button_1_3.setFixedSize(220, 50)
        self.button_1_4.setFixedSize(220, 50)
        self.button_1_5.setFixedSize(220, 50)

        self.label_1 = QLabel("Q1. VGG19", self.frame_1)
        self.label_1.setGeometry(QtCore.QRect(10, 5, 100, 20))
        self.label_1.setFont(font)
        
        # Q1 Image
        self.image_label = QLabel(self.centralwidget)
        self.image_label.setGeometry(QtCore.QRect(325, 25, 250, 250))
        self.image_label.setText("No Image Loaded")
        self.image_label.setFrameShape(QtWidgets.QFrame.Box)
        self.image_label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.image_label.setScaledContents(True)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.result_label = QLabel("Predicted= ", self.centralwidget)
        self.result_label.setGeometry(QtCore.QRect(325, 300, 250, 20))
        self.result_label.setScaledContents(True)
        self.result_label.setWordWrap(True)
        self.layout.addWidget(self.result_label)
        # self.setLayout(self.layout)
        
        
        # Q2
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(620, 25, 250, 400))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.frame_2)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(15, 30, 220, 350))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.button_2_1 = QtWidgets.QPushButton("1. Show Training Images", self.verticalLayoutWidget_1)
        self.button_2_2 = QtWidgets.QPushButton("2. Show Model Structure", self.verticalLayoutWidget_2)
        self.button_2_3 = QtWidgets.QPushButton("3. Show Training Loss", self.verticalLayoutWidget_2)
        self.button_2_4 = QtWidgets.QPushButton("4. Inference", self.verticalLayoutWidget_2)
        self.verticalLayout_2.addWidget(self.button_2_1)
        self.verticalLayout_2.addWidget(self.button_2_2)
        self.verticalLayout_2.addWidget(self.button_2_3)
        self.verticalLayout_2.addWidget(self.button_2_4)
        self.button_2_1.setFixedSize(220, 50)
        self.button_2_2.setFixedSize(220, 50)
        self.button_2_3.setFixedSize(220, 50)
        self.button_2_4.setFixedSize(220, 50)

        self.label_2 = QLabel("Q2. DcGAN", self.frame_2)
        self.label_2.setGeometry(QtCore.QRect(10, 5, 100, 20))
        self.label_2.setFont(font)
        
       