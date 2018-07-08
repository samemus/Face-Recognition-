import cv2
import numpy as np
import sys
from PIL import Image
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal
import datetime
class ShowVideo(QtCore.QObject):
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_recognizer =cv2.face.LBPHFaceRecognizer_create()
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    def startVideo(self):
    #def write_data
        run_video = True
        nb = 20
        while run_video:
            ret,image = self.camera.read()
            only_face = np.array(10)        
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            date = datetime.datetime.now()
            cv2.putText(image, str(date),(370,470), cv2.FONT_ITALIC, 0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(image, "By FALCON TUNISIA",(15,470), cv2.FONT_ITALIC, 0.5,(255,255,255),1,cv2.LINE_AA)
            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),5)
                only_face = gray[y:y+h,x:x+w]            
                cv2.imwrite("data/user"+str(nb)+".jpg", only_face)
            nb = nb + 1         
            cv2.waitKey(1)        
            if nb == 40:            
                self.camera.release()          
                cv2.destroyAllWindows()            
                break
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _= color_swapped_image.shape
            qt_image = QtGui.QImage(color_swapped_image.data,width,height,color_swapped_image.strides[0],QtGui.QImage.Format_RGB888)
            self.VideoSignal.emit(qt_image)
    #def train_data():  
        images = []  
        labels =[]
        face_recognizer =cv2.face.LBPHFaceRecognizer_create()
        for i in range(39):     
            image_pil = Image.open('data/user{}.jpg'.format(i+1)).convert('L')      
            image = np.array(image_pil, 'uint8')    
            faces = face_cascade.detectMultiScale(image)    
            for (x, y, w, h) in faces:    
                images.append(image[y: y + h, x: x + w]) 
                if i<20:       
                    labels.append(1)      
                elif i>19 & i<40 :
                    labels.append(2)
                else:
                    labels.append(3)
                cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])    
                cv2.waitKey(10)
        face_recognizer.update(images, np.array(labels))  
        #f= open("trainer.yml","w+")
        face_recognizer.save('trainer/trainer.yml') 
        cv2.destroyAllWindows()
    #reconnaissance
    def recon_data(self): 
        camera = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_recognizer =cv2.face.LBPHFaceRecognizer_create()
        print("ok")
        face_recognizer.read('trainer/trainer.yml')
        print("i")
        while True:       
            ret, img =camera.read()      
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
            faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),flags=cv2.CASCADE_SCALE_IMAGE)
            for(x, y, w, h) in faces:
                id_user, conf = face_recognizer.predict(gray[y:y+h,x:x+w])       
                cv2.rectangle(img,(x-10,y-10),(x+w+10,y+h+10), (225,255,255),2) 
                if id_user == 1:              
                    name = "moufida"
                    access ="Access allowed"
                elif id_user == 2:              
                    name = "sameher"
                    access ="Access allowed"
                else:
                    name = "indefined"
                    access ="Access not allowed"
                cv2.putText(img,str(name), (x+5,y-15), cv2.FONT_ITALIC, 1.5, 700)
                cv2.putText(img,str(access), (x,y+350), cv2.FONT_ITALIC, 1.5, 25)
                date = datetime.datetime.now()
                cv2.putText(img, str(date),(370,470), cv2.FONT_ITALIC, 0.5,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(img, "By FALCON TUNISIA",(15,470), cv2.FONT_ITALIC, 0.5,(255,255,255),1,cv2.LINE_AA)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            color_swapped_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, _= color_swapped_image.shape
            qt_image = QtGui.QImage(color_swapped_image.data,width,height,color_swapped_image.strides[0],QtGui.QImage.Format_RGB888)
            self.VideoSignal.emit(qt_image)
class ImageViewer(QtWidgets.QWidget):
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0,0, self.image)
    def setImage(self, image):
        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    thread = QtCore.QThread()
    thread.start()
    _translate = QtCore.QCoreApplication.translate
    vid = ShowVideo()
    vid.moveToThread(thread)
    image_viewer = ImageViewer()
    vid.VideoSignal.connect(image_viewer.setImage)
    layout_widget = QtWidgets.QWidget()
    #button start
    push_button1 =QtWidgets.QPushButton('Start')
    push_button1.clicked.connect(vid.recon_data)
    #pour enregistrer des nouvelles captures et realiser des nouveaux modéles on doit décommenter l'instructions ci-dessous et commenter l'instruction ci-dessus 
    #push_button1.clicked.connect(vid.startVideo)

    #calendrier
    calendarWidget = QtWidgets.QCalendarWidget()
    calendarWidget.setGeometry(QtCore.QRect(180, 50, 411, 171))
    calendarWidget.setStyleSheet("alternate-background-color: rgb(204, 204, 204);\n""background-color: rgb(177, 177, 177);")
    calendarWidget.setObjectName("calendarWidget")
    #symbole falcon
    labelImg = QtWidgets.QLabel()
    labelImg.setPixmap(QtGui.QPixmap("test.png"))
    labelImg.setObjectName("labelImg")
    #form vertical de l'interface
    vertical_layout = QtWidgets.QVBoxLayout()
    layout_widget.setLayout(vertical_layout)
    vertical_layout.addWidget(labelImg)
    vertical_layout.addWidget(image_viewer)
    vertical_layout.addWidget(push_button1)
    vertical_layout.addWidget(calendarWidget)
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())
    layout_widget.show()
