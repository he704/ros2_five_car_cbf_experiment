# PRCBF 五车编队避障实验

基于 PRCBF 的五车编队协同避障，包含 ArUco 位姿检测、手势识别、运动控制三个模块。

## 模块与入口
- `aruco_detect`：ArUco 码位姿检测 → `aruco_detect_many_standard.py`
- `gesture_detect`：手势识别 → `lstmmodelrealtime_standard.py`
- `simple_controller`：五车控制 → `five_car_cbf_standard.py`

## 运行

注意：各小车以及中心控制电脑需要连接在同一wifi下

各小车位姿对应话题为/aruco_pose_0`至`/aruco_pose_4

各小车速度指令对应话题为`/cmd_vel_1`至`/cmd_vel_5`

1、修改各小车上订阅的速度话题以及节点名称，如果使用同款亚博智能小车，小车1可按下列操作进行：
   在小车的树莓派上
   
   1、打开终端
   
   2、中断大程序：sh /home/pi/project_demo/raspbot/killprocess.sh
   
   3 、进入容器./docker_ros2.sh
   
   4 、修改驱动代码vim /root/yahboomcar_ws/src/yahboomcar_bringup/launch/bringup.launch.py
   
   5、修改节点名称和订阅的速度话题名称name=’driver_node_1’
                                   remappings=[("/cmd_vel","/cmd_vel_1")]
   
   6、保存：依次输入“： esc wq”
   
   7、回到终端cd /root/yahboomcar_ws
   
   8、编译colcon build
   
   9、启动底盘ros2 launch yahboomcar_bringup bringup.launch.py
   
   依次对五太小车进行该操作即可

在中控电脑上：

2、先运行aruco码检测节点ros2 run aruco_detect aruco_detect_many_standard，
该节点会持续将各小车位姿发送至对应话题

3、再运行小车运动控制节点ros2 run simple_controller five_car_cbf_standard，
小车运动控制节点开始等待手势识别节点给出手势预测信息

4、运行手势识别节点ros2 run gesture_detect lstmmodelrealtime_standard，
按‘s’，开始做手势，触发预测后该节点会将预测落点、触发预测点发送到指定话题，
小车运动控制节点接收到该信息后会分析出目标小车以及所有车的目标位置，之后开始运动

5、所有车到达目标位置后运动结束，如需继续实验，需要“crtl-c”结束小车运动控制节点进程，
再重新启动ros2 run simple_controller five_car_cbf_standard，之后在手势识别节点
按‘s’后即可重新做手势进行实验

## 硬件搭建
1 视觉感知系统

系统配备两台Intel RealSense D455深度相机，分别承担不同的感知任务：

手势相机（顶部俯视）：安装于桌面上方约0.8米处，垂直向下拍摄，用于捕捉老人的手部运动。该相机以1280×720分辨率、30 FPS的帧率采集RGB-D数据，并通过USB线连接笔记本电脑。

定位相机（底部仰视）：平放于地面，垂直向上拍摄，用于检测贴在小车底部的ArUco二维码标记位姿。该相机同样以1280×720@30 FPS运行，通过USB线连接笔记本电脑。

2 移动机器人平台

实验采用五辆Yahboom RASPBOTV2 ROS小车作为执行单元。每辆小车配备麦克纳姆轮，可在桌面上实现全向运动。小车底部中心粘贴有4×4规格的ArUco二维码（ID分别为0–4，物理尺寸71 mm），供底部定位相机识别。

每辆小车上搭载树莓派5，运行ROS 2 Humble系统。树莓派通过Wi-Fi与中央电脑通信，接收来自电脑的速度指令话题，底盘驱动节点将速度指令转换为电机控制信号执行。

3 实验场地

实验在标准室内环境下进行，桌面工作区域尺寸约为1.36 m × 0.7m，桌面材质采用透明亚克力板（厚度10 mm），允许底部相机透过桌面观测到所有小车的标记，实现无遮挡的全局定位，老人座位位于于桌面一侧，与五辆小车的初始摆放区域相对。
