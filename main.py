# Author: Adwit Singh Kochar

import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import datetime, time
import os

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s
def start():
    # KEY = 0

    f = open("a","r").readlines()
    foldername = 'newpics' + str(len(f)+1)
    if(os.path.isdir(foldername) == False):
        os.system("mkdir "+foldername)

    USE_WEBCAM = True 

    # parameters for loading data and images
    emotion_model_path = './models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []
    l = time.time()

    # starting video streaming

    cv2.namedWindow('window_frame')
    #video_capture = cv2.VideoCapture(0)

    # Select video or webcam feed
    global cap
    if (USE_WEBCAM == True):
        cap = cv2.VideoCapture(0) # Webcam source
    else:
        cap = cv2.VideoCapture('./demo/video.mp4') # Video file source
    i = 0
    while cap.isOpened(): # True:
        flag = 1        
        ret, bgr_image = cap.read()
        #bgr_image = video_capture.read()[1]
        m = 0
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        o = bgr_image

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:
            m = 1
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
                flag = 0

            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
                flag = 0

            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
                if flag == 1 and emotion_probability*100 >=60 :
                    flag = 1
                else:
                    flag = 0

            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
                if flag == 1 and emotion_probability*100 >=60:
                    flag = 1
                else:
                    flag = 0

            else:
                color = emotion_probability * np.asarray((0, 255, 0))
                flag = 0


            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)

        if time.time() - l >= 3 and flag == 1 and m == 1:
            cv2.imwrite(foldername + '/pic'+str(i)+'.png',o)
            i+=1
            l = time.time()

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


def stop():
    global cap
    cap.release()

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.setEnabled(True)
        Form.resize(400, 300)
        self.label = QtGui.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(110, 10, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.plainTextEdit = QtGui.QPlainTextEdit(Form)
        self.plainTextEdit.setEnabled(True)
        self.plainTextEdit.setGeometry(QtCore.QRect(180, 70, 51, 31))
        self.plainTextEdit.setObjectName(_fromUtf8("plainTextEdit"))
        self.plainTextEdit_3 = QtGui.QPlainTextEdit(Form)
        self.plainTextEdit_3.setGeometry(QtCore.QRect(300, 70, 51, 31))
        self.plainTextEdit_3.setObjectName(_fromUtf8("plainTextEdit_3"))
        self.radioButton = QtGui.QRadioButton(Form)
        self.radioButton.setEnabled(True)
        self.radioButton.setGeometry(QtCore.QRect(50, 80, 114, 16))
        self.radioButton.setStatusTip(_fromUtf8(""))
        self.radioButton.setChecked(False)
        self.radioButton.setObjectName(_fromUtf8("radioButton"))
        self.plainTextEdit_2 = QtGui.QPlainTextEdit(Form)
        self.plainTextEdit_2.setGeometry(QtCore.QRect(240, 70, 51, 31))
        self.plainTextEdit_2.setObjectName(_fromUtf8("plainTextEdit_2"))
        self.pushButton = QtGui.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(80, 150, 95, 27))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton_2 = QtGui.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(220, 150, 95, 27))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_3 = QtGui.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(80, 190, 95, 27))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton_4 = QtGui.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(220, 190, 95, 27))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))

        self.retranslateUi(Form)
        QtCore.QObject.connect(self.pushButton, QtCore.SIGNAL(_fromUtf8("clicked()")), start)
        QtCore.QObject.connect(self.pushButton_2, QtCore.SIGNAL(_fromUtf8("clicked()")), stop)
      #  QtCore.QObject.connect(self.pushButton_3, QtCore.SIGNAL(_fromUtf8("clicked()")), view_photos)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.label.setText(_translate("Form", "MOMENTS", None))
        self.plainTextEdit.setPlainText(_translate("Form", "HH", None))
        self.plainTextEdit_3.setPlainText(_translate("Form", "SS", None))
        self.radioButton.setText(_translate("Form", "Set Timer", None))
        self.plainTextEdit_2.setPlainText(_translate("Form", "MM", None))
        self.pushButton.setText(_translate("Form", "Start", None))
        self.pushButton_2.setText(_translate("Form", "Stop", None))
        self.pushButton_3.setText(_translate("Form", "View Photos", None))
        self.pushButton_4.setText(_translate("Form", "Reset", None))


if __name__ == "__main__":
    import sys
    os.system("ls > a")
    app = QtGui.QApplication(sys.argv)
    Form = QtGui.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

