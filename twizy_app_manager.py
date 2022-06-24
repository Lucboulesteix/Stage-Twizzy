from tkinter import E
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,QCheckBox, QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5 import QtCore
from PyQt5 import QtGui

from roslaunch.parent import ROSLaunchParent
import sys
import subprocess
import roslaunch
import time
import rosnode

#interface callbacks
def rosrecord_button_clicked():
    global rosrecordStarted, launch_realsense
    if not(rosrecordStarted):

        file_path ='twizy_record_data.launch'
        uuid = roslaunch.rlutil.get_or_generate_uuid('TwizyROScore', False)
        launch_realsense = roslaunch.parent.ROSLaunchParent(uuid, [file_path])
        launch_realsense.start()

        rosrecord_button.setStyleSheet("background-color: green")
        rosrecord_button.setText("Stop recording")
        rosrecordStarted = True
    else:
        launch_realsense.shutdown()
        rosrecord_button.setStyleSheet("background-color: lightgray")
        rosrecord_button.setText("Start recording")
        rosrecordStarted = False

def realsense_camera_clicked():
    global realsenseStarted, launch_realsense
    if not(realsenseStarted):
        file_path ='twizy_realsense_launch.launch'
        uuid = roslaunch.rlutil.get_or_generate_uuid('TwizyROScore', False)
        launch_realsense = roslaunch.parent.ROSLaunchParent(uuid, [file_path])
        launch_realsense.start()
        realsenseStarted = True
        realsense_camera_button.setStyleSheet("background-color: green")
    else:
        launch_realsense.shutdown()
        realsenseStarted = False
        realsense_camera_button.setStyleSheet("background-color: lightgray")

def velodyne_clicked():
    global velodyneStarted, launch_velodyne
    if not(velodyneStarted):
        file_path ='twizy_velodyne_launch.launch'
        uuid = roslaunch.rlutil.get_or_generate_uuid('TwizyROScore', False)
        launch_velodyne = roslaunch.parent.ROSLaunchParent(uuid, [file_path])
        launch_velodyne.start()
        velodyneStarted = True
        velodyne_button.setStyleSheet("background-color: green")
    else:
        launch_velodyne.shutdown()
        velodyneStarted = False
        velodyne_button.setStyleSheet("background-color: lightgray")

def graph_clicked():
    global graphStarted, launch_graph
    if not(graphStarted):
        file_path ='twizy_tools.launch'
        uuid = roslaunch.rlutil.get_or_generate_uuid('TwizyROScore', False)
        launch_graph = roslaunch.parent.ROSLaunchParent(uuid, [file_path])
        launch_graph.start()
        graphStarted = True
        graph_button.setStyleSheet("background-color: green")
    else:
        launch_graph.shutdown()
        graphStarted = False
        graph_button.setStyleSheet("background-color: lightgray")

def object_detector_clicked():
    global objDetectStarted, launch_objDetect, objDetect_process
    if not(objDetectStarted):
        objDetect_process = subprocess.Popen(['sh /home/srodri2/Bureau/twizy_app/object_detector/catkin_ws/src/detectors/src/run_realsense_objDect_noeud.sh'], shell=True)
        objDetectStarted = True
        objDetect_button.setStyleSheet("background-color: green")
    else:
        liste = rosnode.get_node_names()
        indices = [i for i, s in enumerate(liste) if 'realsense_object_detector' in s]
        if len(indices)==1:
            subprocess.call(['rosnode','kill',liste[indices[0]]])
            print('[INFO] ROS node '+liste[indices[0]]+ ' terminated')
        else:
            print('[INFO] : Could not find node to terminate')
        objDetectStarted = False
        objDetect_button.setStyleSheet("background-color: lightgray")

def display_clicked():
    global displayStarted, launch_rviz
    if not(displayStarted):
        file_path ='twizy_rviz_launch.launch'
        uuid = roslaunch.rlutil.get_or_generate_uuid('TwizyROScore', False)
        launch_rviz = roslaunch.parent.ROSLaunchParent(uuid, [file_path])
        launch_rviz.start()
        displayStarted = True
        display_button.setStyleSheet("background-color: green")
    else:
        launch_rviz.shutdown()
        displayStarted = False
        display_button.setStyleSheet("background-color: lightgray")

def quit_button_clicked():
    print('Close application')
    sys.exit()


#GUI launcher application
app = QApplication(["Twizy ROS"])
window = QWidget()

sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
sizePolicy.setHeightForWidth(True)
window.setSizePolicy(sizePolicy)

window.setWindowTitle('Twizy ROS application manager')
window.setWindowIcon(QtGui.QIcon("./icon.png"))
window.resize(500, 500)
window.move(300, 300)
Gral_layout = QVBoxLayout()
lin1_layout = QHBoxLayout()
lin2_layout = QHBoxLayout()

global realsenseStarted
realsenseStarted = False
realsense_camera_button = QPushButton('Realsense Cameras')
realsense_camera_button.clicked.connect(realsense_camera_clicked)
realsense_camera_button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

global velodyneStarted
velodyneStarted = False
velodyne_button = QPushButton('Velodyne LiDAR')
velodyne_button.clicked.connect(velodyne_clicked)
velodyne_button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
velodyne_button.resize(100,100)

global displayStarted
displayStarted = False
display_button = QPushButton('Show data')
display_button.clicked.connect(display_clicked)
display_button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
velodyne_button.resize(100,100)

global objDetectStarted
objDetectStarted = False
objDetect_button = QPushButton('Detect objects')
objDetect_button.clicked.connect(object_detector_clicked)
objDetect_button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
objDetect_button.resize(100,100)

global graphStarted
graphStarted = False
graph_button = QPushButton('Show graph')
graph_button.clicked.connect(graph_clicked)
graph_button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
graph_button.resize(100,100)

global rosrecordStarted
rosrecordStarted = False
rosrecord_button = QPushButton('Start recording')
rosrecord_button.clicked.connect(rosrecord_button_clicked)
rosrecord_button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

quit_button    = QPushButton('Quit')
quit_button.clicked.connect(quit_button_clicked)
quit_button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

#layout.addStretch()
lin1_layout.addWidget(realsense_camera_button)
lin1_layout.addWidget(velodyne_button)
lin1_layout.addWidget(objDetect_button)
lin1_layout.addWidget(display_button)

lin2_layout.addWidget(graph_button)
lin2_layout.addWidget(rosrecord_button)
lin2_layout.addWidget(quit_button)

Gral_layout.addLayout(lin1_layout)
Gral_layout.addLayout(lin2_layout)
#layout.addStretch()
window.setLayout(Gral_layout)

window.show()
sys.exit(app.exec())
