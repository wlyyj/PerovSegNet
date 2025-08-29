import os
from datetime import datetime
import json
import cv2
import torch
from PyQt5.QtCore import pyqtSlot
from PySide6.QtGui import QIcon
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, QDir
from ultralytics import YOLO


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_gui()
        self.model = None
        self.timer = QtCore.QTimer()
        self.timer1 = QtCore.QTimer()
        self.cap = None
        self.video = None
        self.file_path = None
        self.base_name = None
        self.timer1.timeout.connect(self.video_show)

    def init_gui(self):
        self.folder_path = "D:/ultralytics-main"  # 自定义修改：设置模型路径
        self.setFixedSize(1300, 650)
        self.setWindowTitle('Perovskite And Lead Iodide Automatic System')  # 自定义修改：设置窗口名称
        self.setWindowIcon(QIcon("D:/ultralytics-main/runs/segment/train2/train_batch2.jpg"))  # 自定义修改：设置窗口图标
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # 界面上半部分： 视频框
        # videoBox = QtWidgets.QGroupBox(self)
        # videoBox.setStyleSheet('QGroupBox {border: 0px solid #D7E2F9;}')
        # videoLayout = QtWidgets.QVBoxLayout(videoBox)
        # main_layout.addWidget(videoBox)
        topLayout = QtWidgets.QHBoxLayout()
        self.oriVideoLabel = QtWidgets.QLabel(self)
        self.detectlabel = QtWidgets.QLabel(self)
        self.oriVideoLabel.setFixedSize(530, 400)
        self.detectlabel.setFixedSize(530, 400)
        self.oriVideoLabel.setStyleSheet('border: 2px solid #ccc; border-radius: 10px; margin-top:75px;')
        self.detectlabel.setStyleSheet('border: 2px solid #ccc; border-radius: 10px; margin-top: 75px;')
        # 960 540  1920 960
        topLayout.addWidget(self.oriVideoLabel)
        topLayout.addWidget(self.detectlabel)
        main_layout.addLayout(topLayout)

        # 界面下半部分： 输出框 和 按钮
        groupBox = QtWidgets.QGroupBox(self)
        groupBox.setStyleSheet('QGroupBox {border: 0px solid #D7E2F9;}')
        bottomLayout = QtWidgets.QHBoxLayout(groupBox)
        main_layout.addWidget(groupBox)
        btnLayout = QtWidgets.QHBoxLayout()
        btn1Layout = QtWidgets.QVBoxLayout()
        btn2Layout = QtWidgets.QVBoxLayout()
        btn3Layout = QtWidgets.QVBoxLayout()

        # 创建日志打印文本框
        self.outputField = QtWidgets.QTextBrowser()
        self.outputField.setFixedSize(530, 180)
        # self.outputField.setStyleSheet('font-size: 13px; font-family: "Microsoft YaHei"; background-color: #f0f0f0;'
        #                                ' border: 2px solid #ccc; border-radius: 10px;')
        self.outputField.setStyleSheet("""
            QTextBrowser {  
                border: 2px solid gray;  
                border-radius: 10px;  
                padding: 7px;  
                background-color: #f0f0f0;  
                font-size: 14px;
            }  
            QTextBrowser QScrollBar:vertical {
                width: 20px; 
            }
            QTextBrowser QScrollBar::handle:vertical {  
                background: #cccccc;  
                min-height: 20px;  
                border-radius: 5px;
            }  

            QTextBrowser QScrollBar::add-line:vertical {  
                border: 1px solid #999999;  
                background: #ffffff;  
                height: 16px;  
                subcontrol-position: bottom;  
                subcontrol-origin: margin;  
            }  

            QTextBrowser QScrollBar::sub-line:vertical {  
                border: 1px solid #999999;  
                background: #ffffff;  
                height: 16px;  
                subcontrol-position: top;  
                subcontrol-origin: margin;  
            }  

            QTextBrowser QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {  
                background: none;  
            }  

            QTextBrowser QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {  
                border: 1px solid grey;  
                width: 6px;  
                height: 6px;  
                background: white;  
            }  

            QTextBrowser QScrollBar::up-arrow:vertical:hover, QScrollBar::down-arrow:vertical:hover {  
                background: #cccccc;  
            }  

            QTextBrowser QScrollBar::add-page:vertical:hover, QScrollBar::sub-page:vertical:hover {  
                background: #eeeeee;  
            }
        """)
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Please select a model file')

        # 创建选择模型下拉框
        selectModel_layout = QtWidgets.QHBoxLayout()
        self.selectModel = QtWidgets.QComboBox()
        self.selectModel.setFixedSize(130, 50)
        self.selectModel.setStyleSheet("""  
            QComboBox {  
                border: 2px solid gray;  
                border-radius: 10px;  
                padding: 5px;  
                background-color: #f0f0f0;  
                font-size: 14px;  
            }  

            QComboBox QAbstractItemView {  
                background-color: #ffffff;  /* 下拉列表的背景色 */  
                selection-background-color: #c0c0c0;  /* 选中项的背景色 */  
                selection-color: black;  /* 选中项的文字颜色 */  
            }  

            QComboBox::drop-down {  
                subcontrol-origin: padding;  
                subcontrol-position: top right;  
                width: 20px;  
                border-width: 1px;  
                border-style: solid;  
                border-color: lightgray;  
                border-radius: 8px;  
                background: white;  
            }  

            QComboBox::down-arrow {  
                image: none;  
                width: 7px;  
                height: 7px;  
                background-color: black;  
            }  

            QComboBox::down-arrow:on {  
                background-color: darkgray;  
            }
        """)
        # 遍历文件夹并添加文件名到下拉框
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.pt'):  # 确保是文件且后缀为.pt
                base_name = os.path.splitext(filename)[0]
                self.selectModel.addItem(base_name)
        # 添加加载模型按钮
        self.loadModel = QtWidgets.QPushButton('🔄️load model')
        self.loadModel.setFixedSize(125, 50)
        self.loadModel.setStyleSheet("""
            QPushButton {  
                background-color: white; /* 正常状态下的背景颜色 */  
                border: 2px solid gray;  /* 正常状态下的边框 */  
                border-radius: 10px;  
                padding: 5px;
                font-size: 14px; 
            }  

            QPushButton:hover {  
                background-color: #f0f0f0;  /* 悬停状态下的背景颜色 */  
            }  
        """)
        self.loadModel.clicked.connect(self.load_model)
        selectModel_layout.addWidget(self.selectModel)
        selectModel_layout.addWidget(self.loadModel)

        # 创建一个置信度阈值滑动条
        self.con_label = QtWidgets.QLabel('Confidence threshold', self)
        self.con_label.setStyleSheet('font-size: 14px; font-family: "Microsoft YaHei";')
        # 创建一个QSlider，范围从0到99（代表0.01到0.99）
        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(1)  # 0.01
        self.slider.setMaximum(99)  # 0.99
        self.slider.setValue(50)  # 0.5
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setFixedSize(170, 30)
        # 创建一个QDoubleSpinBox用于显示和设置滑动条的值
        self.spinbox = QtWidgets.QDoubleSpinBox(self)
        self.spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinbox.setMinimum(0.01)
        self.spinbox.setMaximum(0.99)
        self.spinbox.setSingleStep(0.01)
        self.spinbox.setValue(0.5)
        self.spinbox.setDecimals(2)
        self.spinbox.setFixedSize(60, 30)
        self.spinbox.setStyleSheet('border: 2px solid gray; border-radius: 10px; '
                                   'padding: 5px; background-color: #f0f0f0; font-size: 14px;')
        self.confudence_slider = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        self.confudence_slider.setFixedSize(250, 64)
        layout.addWidget(self.con_label)
        hlayout.addWidget(self.slider)
        hlayout.addWidget(self.spinbox)
        layout.addLayout(hlayout)
        self.confudence_slider.setLayout(layout)
        self.confudence_slider.setEnabled(False)
        # 连接信号和槽
        self.slider.valueChanged.connect(self.updateSpinBox)
        self.spinbox.valueChanged.connect(self.updateSlider)

        # 执行预测按钮
        self.start_detect = QtWidgets.QPushButton('🔍Start')
        self.start_detect.setFixedSize(100, 50)
        self.start_detect.setStyleSheet("""
            QPushButton {  
                background-color: white; /* 正常状态下的背景颜色 */  
                border: 2px solid gray;  /* 正常状态下的边框 */  
                border-radius: 10px;  
                padding: 5px;
                font-size: 14px;
            }  

            QPushButton:hover {  
                background-color: #f0f0f0;  /* 悬停状态下的背景颜色 */  
            }  
        """)
        self.start_detect.clicked.connect(self.show_detect)
        self.start_detect.setEnabled(False)

        # 文件上传按钮
        self.openImageBtn = QtWidgets.QPushButton('🖼️Upload File')
        self.openImageBtn.setFixedSize(125, 65)
        self.openImageBtn.setStyleSheet("""
            QPushButton {  
                background-color: white; /* 正常状态下的背景颜色 */  
                border: 2px solid gray;  /* 正常状态下的边框 */  
                border-radius: 10px;  
                padding: 5px;
                font-size: 14px; 
                margin-bottom: 15px;
            }  

            QPushButton:hover {  
                background-color: #f0f0f0;  /* 悬停状态下的背景颜色 */  
            }  
        """)
        self.openImageBtn.clicked.connect(self.upload_file)
        self.openImageBtn.setEnabled(False)

        # 停止检测按钮
        self.stopDetectBtn = QtWidgets.QPushButton('🛑Stop')
        self.stopDetectBtn.setFixedSize(100, 50)
        self.stopDetectBtn.setStyleSheet("""
            QPushButton {  
                background-color: white; /* 正常状态下的背景颜色 */  
                border: 2px solid gray;  /* 正常状态下的边框 */  
                border-radius: 10px;  
                padding: 5px;
                font-size: 14px; 
            }  

            QPushButton:hover {  
                background-color: #f0f0f0;  /* 悬停状态下的背景颜色 */  
            }  
        """)
        self.stopDetectBtn.setEnabled(False)
        self.stopDetectBtn.clicked.connect(self.stop_detect)

        # 布局整理
        self.operate = QtWidgets.QWidget()
        btn1Layout.addLayout(selectModel_layout)
        btn1Layout.addWidget(self.confudence_slider)
        btn2Layout.addWidget(self.openImageBtn)
        btn2Layout.addWidget(self.start_detect)
        btn3Layout.addWidget(self.stopDetectBtn)
        bottomLayout.addWidget(self.outputField)
        self.operate.setFixedSize(530, 200)
        self.operate.setLayout(btnLayout)
        btnLayout.addLayout(btn1Layout)
        btnLayout.addLayout(btn2Layout)
        btnLayout.addLayout(btn3Layout)
        bottomLayout.addWidget(self.operate)
        bottomLayout.setContentsMargins(0, 30, 0, 10)
        self.set_background_image('bg.jpg')  # 自定义修改：设置窗口背景图
        self.ini_labels()

    @pyqtSlot(int)
    def updateSpinBox(self, value):
        # 将slider的值转换为0.01到0.99之间的浮点数
        self.spinbox.setValue(value * 0.01)

    @pyqtSlot(float)
    def updateSlider(self, value):
        # 将spinbox的值转换为1到99之间的整数
        self.slider.setValue(int(value * 100))

    def set_background_image(self, image_path):
        # 设置样式表来应用背景图片
        self.setStyleSheet(f"QMainWindow {{ border-image: url('{image_path}'); }}")

    def upload_file(self):
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Please select the detection file')
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setDirectory(QDir("./jpg_file"))  # 自定义设置推理图片/视频路径。
        file_path, file_type = file_dialog.getOpenFileName(self, "Select detection file", filter='*.jpg *.mp4')
        self.file_path = file_path
        if file_path:
            if file_path.endswith('.jpg'):
                # 在这里添加加载图片的逻辑
                pixmap = QtGui.QPixmap(file_path)
                self.oriVideoLabel.setPixmap(pixmap)
                self.oriVideoLabel.setScaledContents(True)
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Picture uploaded successfully: {file_path}')
            if file_path.endswith('.mp4'):
                # 在这里添加加载视频的逻辑
                self.video = cv2.VideoCapture(file_path)
                ret, frame = self.video.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(frame))
                self.oriVideoLabel.setScaledContents(True)
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Video uploaded successfully: {file_path}')
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Please reselect the detection file！')

    def show_detect(self):
        if self.model is not None:
            if self.file_path:
                if self.file_path.endswith('.jpg'):
                    self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Start detection......')
                    result = self.model(self.file_path, device='cuda',
                                        conf=self.spinbox.value()) if torch.cuda.is_available() \
                        else self.model(self.file_path, device='cpu', conf=self.spinbox.value())
                    # result[0].names[0] = "道路积水"
                    frame = cv2.cvtColor(result[0].plot(), cv2.COLOR_RGB2BGR)
                    # 将图像数据转为QImage格式
                    frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3,
                                         QtGui.QImage.Format_RGB888)
                    self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(frame))
                    self.detectlabel.setScaledContents(True)
                    self.output_json(result)
                    self.outputField.append(
                        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Image detection completed！: {self.file_path}')
                if self.file_path.endswith('.mp4'):
                    self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Start detection......')
                    self.video = cv2.VideoCapture(self.file_path)
                    fps = self.video.get(cv2.CAP_PROP_FPS)
                    self.timer1.start(int(1 / fps))
            else:
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Please reselect the detection file！')
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Please load the model first！')

    def video_show(self):
        ret, frame = self.video.read()
        if ret:
            if self.model is not None:
                frame = cv2.resize(frame, (448, 352))
                frame1 = self.model(frame, imgsz=[448, 352], device='cuda',
                                    conf=self.spinbox.value()) if torch.cuda.is_available() \
                    else self.model(frame, imgsz=[448, 352], device='cpu', conf=self.spinbox.value())
                temp = frame1
                frame1 = cv2.cvtColor(frame1[0].plot(), cv2.COLOR_RGB2BGR)
                frame1 = QtGui.QImage(frame1.data, frame1.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(frame1))
                self.detectlabel.setScaledContents(True)
                self.output_json(temp)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(frame))
            self.oriVideoLabel.setScaledContents(True)
        else:
            self.timer1.stop()
            self.outputField.append(
                f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Video detection completed！: {self.file_path}')
            self.ini_labels()
            self.video.release()
            self.video = None
            self.file_path = None

    def output_json(self, result):
        specified_items = []
        for i in range(len(result[0].boxes.cls)):
            if result[0].boxes.conf[i] > self.spinbox.value():
                specified_items.append({
                    'name': result[0].names[result[0].boxes.cls[i].item()],
                    'confidence': result[0].boxes.conf[i].item(),
                    'x': result[0].boxes.xyxy[i][0].item(),
                    'y': result[0].boxes.xyxy[i][1].item(),
                    'width': result[0].boxes.xyxy[i][2].item(),
                    'height': result[0].boxes.xyxy[i][3].item()
                })
                self.outputField.append(
                    f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - '
                    f'⚠️alarm！: {self.base_name}————{json.dumps(specified_items, ensure_ascii=False, indent=4)}')
                specified_items.clear()

    def ini_labels(self):
        img = QtGui.QImage(500, 500, QtGui.QImage.Format_ARGB32)
        img.fill(QtGui.qRgba(0, 0, 0, 0))
        # img = cv2.cvtColor(np.zeros((500, 500), np.uint8), cv2.COLOR_BGR2RGB)
        # img = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        # img.color(QtGui.QColor(255, 255, 255))
        self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(img))
        self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(img))

    def load_model(self):
        filename = self.selectModel.currentText()
        full_path = os.path.join(self.folder_path, filename + '.pt')
        self.base_name = os.path.splitext(os.path.basename(full_path))[0]
        if full_path.endswith('.pt'):
            self.model = YOLO(full_path)
            self.start_detect.setEnabled(True)
            self.stopDetectBtn.setEnabled(True)
            self.openImageBtn.setEnabled(True)
            self.confudence_slider.setEnabled(True)
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Model loaded successfully: {filename}')
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Please select a confidence threshold')
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Please reselect the model file！')
            print("Reselect model")

    def stop_detect(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.timer1.isActive():
            self.timer1.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video = None
        self.ini_labels()
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Detecting interrupts！')
        self.file_path = None

    def video_show2(self):
        ret, frame = self.video.read()
        if ret:
            if self.model is not None:
                frame = cv2.resize(frame, (448, 352))
                frame1 = self.model(frame, imgsz=[448, 352], device='cuda',
                                    conf=self.spinbox.value()) if torch.cuda.is_available() \
                    else self.model(frame, imgsz=[448, 352], device='cpu', conf=self.spinbox.value())
                self.output_json(frame1)
                frame1 = cv2.cvtColor(frame1[0].plot(), cv2.COLOR_RGB2BGR)
                frame1 = QtGui.QImage(frame1.data, frame1.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                pic_path = './runs/output/output_image.jpg'
                frame1.save(pic_path)
                frame2 = self.model2(pic_path, imgsz=[448, 352], device='cuda',
                                     conf=self.spinbox.value()) if torch.cuda.is_available() \
                    else self.model2(pic_path, imgsz=[448, 352], device='cpu', conf=self.spinbox.value())
                self.output_json(frame2)
                os.remove(pic_path)
                frame2 = cv2.cvtColor(frame2[0].plot(), cv2.COLOR_RGB2BGR)
                frame2 = QtGui.QImage(frame2.data, frame2.shape[1], frame2.shape[0], QtGui.QImage.Format_RGB888)
                self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(frame2))
                self.detectlabel.setScaledContents(True)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(frame))
            self.oriVideoLabel.setScaledContents(True)
        else:
            self.timer1.stop()
            self.outputField.append(
                f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Video detection completed！: {self.file_path}')
            self.ini_labels()
            self.video.release()
            self.video = None
            self.file_path = None


if __name__ == '__main__':
    app = QtWidgets.QApplication()
    window = MyWindow()
    window.show()
    app.exec()

