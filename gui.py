import sys
import time

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import test3


class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()

        self.resize(1000, 1000)
        self.setWindowTitle("图像分割")

        self.label = QLabel(self)
        self.label.setFixedSize(800, 800)
        self.label.move(160, 160)

        self.pic_name=''

        self.label.setStyleSheet("QLabel{background:#3a96dd;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )

        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.move(30, 30)
        btn.clicked.connect(self.openimage)
        btn.setFixedSize(160, 60)
        btn.setStyleSheet(
            '''QPushButton{background:#e54437;border-radius:5px;}QPushButton:hover{background:#880604;}''')

        self.bt2 = QPushButton('分割', self)
        self.bt2.move(200,30)
        self.bt2.clicked.connect(self.divide)
        self.bt2.setFixedSize(160,60)
        self.bt2.setStyleSheet(
            '''QPushButton{background:#57da65;border-radius:5px;}QPushButton:hover{background:#1f7f12;}''')

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        self.pic_name=imgName

    def divide(self):
        imgName=self.pic_name
        test3.main(imgName)
        jpg1 = QtGui.QPixmap(str(imgName).split('/')[-1]).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg1)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())