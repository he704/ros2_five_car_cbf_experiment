"""
五车CBF安全位置控制器（完整版）- 增强桌边安全约束
功能：控制5小车安全到达指定位置，包含完整的概率CBF屏障证书和桌边安全约束
基于四车版本修改，支持五车协同控制和边界避障
"""

import rclpy
from rclpy.node import Node
import math
import numpy as np
from scipy.optimize import minimize
from geometry_msgs.msg import Pose2D, Twist
import csv
import os
from datetime import datetime

class CBFFiveCarController(Node):
    def __init__(self):
        super().__init__('cbf_five_car_controller')
        
        # ==================== 控制参数 ====================
        # 小车目标位置和角度（五车配置）
        self.target_1_x = None
        self.target_1_y = None
        self.target_1_z = 0  # 面朝右
        
        self.target_2_x = None
        self.target_2_y = None
        self.target_2_z = 0  # 面朝右

        self.target_3_x = None
        self.target_3_y = None
        self.target_3_z = 0  # 面朝右
        
        self.target_4_x = None
        self.target_4_y = None#0.1
        self.target_4_z = 0  # 面朝右
        
        # 新增第五辆小车目标位置
        self.target_5_x = None
        self.target_5_y = None
        self.target_5_z = 0  # 面朝右

        ##
        self.target_x=None
        self.target_y=None

        self.trigger_x=None
        self.trigger_y=None
        self.trigger_z=None

        self.targetcar=None
        self.offset_x=44#38
        self.offset_y=31#38

        
        # ==================== 桌面边界参数（新增） ====================
        # 桌面边界定义（基于Robotarium标准尺寸）
        self.table_boundaries = {
            'x_min': -0.66,    # 桌面左边界
            'x_max': 0.66,     # 桌面右边界
            'y_min': -0.37,   # 桌面下边界
            'y_max': 0.31     # 桌面上边界
        }
        
        # 边界安全距离（小车与边界的最小安全距离）
        self.boundary_safety_margin = 0.0  # 8cm安全边距
        
        # 边界屏障增益（新增）
        self.boundary_gain = 800  # 边界约束增益
        
        # PID增益（新增第五辆小车）
        # self.kp_1_x, self.kp_1_y, self.kp_1_z = 1.0, 1.0, 1.0
        # self.kp_2_x, self.kp_2_y, self.kp_2_z = 1.0, 1.0, 1.0
        # self.kp_3_x, self.kp_3_y, self.kp_3_z = 1.0, 1.0, 1.0
        # self.kp_4_x, self.kp_4_y, self.kp_4_z = 1.0, 1.0, 1.0
        # self.kp_5_x, self.kp_5_y, self.kp_5_z = 1.0, 1.0, 1.0  # 新增

        self.kp_1_x, self.kp_1_y, self.kp_1_z = 1.0, 1.0, 0.5
        self.kp_2_x, self.kp_2_y, self.kp_2_z = 1.0, 1.0, 0.5
        self.kp_3_x, self.kp_3_y, self.kp_3_z = 1.0, 1.0, 0.5
        self.kp_4_x, self.kp_4_y, self.kp_4_z = 1.0, 1.0, 0.5
        self.kp_5_x, self.kp_5_y, self.kp_5_z = 1.0, 1.0, 0.5  # 新增
        
        # 速度限制
        self.max_speed_xy = 0.12    # x,y最大速度
        self.max_speed_z = 0.1     # 角速度限制 0.45
        
        # 停止容差
        self.stop_tolerance_xy = 0.07#0.07
        self.stop_tolerance_z = 0.18#0.09
        
        # ==================== CBF参数 ====================
        self.safety_radius = 0.15      # 安全半径 18cm
        self.safety_radius_boundary = 0.0
        self.safety_radius_boundary_start = 0.05
        self.barrier_gain = 500        # 屏障增益γ
        self.confidence = 0.5          # 置信水平 100%
        
        # 不确定性参数
        self.x_rand_span = 0.03       # 位置误差 ±3cm
        self.u_rand_span = 0.005      # 速度误差 ±5mm/s
        
        # ==================== 状态变量 ====================
        # 新增第五辆小车状态变量
        self.current_1_x = self.current_1_y = self.current_1_z = None
        self.current_2_x = self.current_2_y = self.current_2_z = None
        self.current_3_x = self.current_3_y = self.current_3_z = None
        self.current_4_x = self.current_4_y = self.current_4_z = None
        self.current_5_x = self.current_5_y = self.current_5_z = None  # 新增
        
        self.state_1 = "等待二维码"
        self.state_2 = "等待二维码"
        self.state_3 = "等待二维码"
        self.state_4 = "等待二维码"
        self.state_5 = "等待二维码"  # 新增
        self.has_reached_target = False
        
        # 边界违规记录（新增）
        self.boundary_violation_count = 0
        self.last_boundary_warning_time = 0
        self.targetset=False
        
        # ==================== 数据记录初始化 ====================
        self.data_recording_enabled = True  # 数据记录开关
        self.data_record_interval = 0.05     # 数据记录间隔（秒），配合20Hz控制频率，每2次记录一次
        self.last_record_time = 0.0         # 上次记录时间
        # 在其下方添加
        self.recording_start_time = None    # 运动开始时间（用于生成相对时间戳）
        
        # 创建数据存储目录
        self.data_dir = os.path.join(os.path.expanduser('~'), 'robot_trajectory_data')
        if self.data_recording_enabled:
            os.makedirs(self.data_dir, exist_ok=True)
            
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = os.path.join(self.data_dir, f'trajectory_data_{timestamp}.csv')
        
        # CSV文件头
        self.csv_header = [
            'timestamp',           # 时间戳（秒）
            'car1_x', 'car1_y',    # 小车1坐标
            'car2_x', 'car2_y',    # 小车2坐标
            'car3_x', 'car3_y',    # 小车3坐标
            'car4_x', 'car4_y',    # 小车4坐标
            'car5_x', 'car5_y',    # 小车5坐标
            'target_car',          # 被选中的目标小车
            'target_x', 'target_y', # 手势目标位置
            'state'                # 系统状态
        ]
        
        # 打开CSV文件并写入表头
        if self.data_recording_enabled:
            try:
                self.csv_file = open(self.csv_filename, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                self.csv_writer.writerow(self.csv_header)
                self.get_logger().info(f"📝 数据记录已开启，文件路径: {self.csv_filename}")
            except Exception as e:
                self.get_logger().error(f"❌ 无法创建数据记录文件: {e}")
                self.data_recording_enabled = False
        
        # ==================== ROS设置 ====================
        # 订阅二维码位姿（五车）
        self.pose_sub_1 = self.create_subscription(
            Pose2D, '/aruco_pose_0', self.pose_callback_1, 1)
        self.pose_sub_2 = self.create_subscription(
            Pose2D, '/aruco_pose_1', self.pose_callback_2, 1)
        self.pose_sub_3 = self.create_subscription(
            Pose2D, '/aruco_pose_2', self.pose_callback_3, 1)
        self.pose_sub_4 = self.create_subscription(
            Pose2D, '/aruco_pose_3', self.pose_callback_4, 1)
        self.pose_sub_5 = self.create_subscription(  # 新增
            Pose2D, '/aruco_pose_4', self.pose_callback_5, 1)
        self.pose_sub_6 = self.create_subscription(  # 新增
             Twist, '/gesture_twist_data', self.pose_callback_6, 1)
        # 发布控制指令（五车）
        self.cmd_pub_1 = self.create_publisher(Twist, '/cmd_vel_1', 1)
        self.cmd_pub_2 = self.create_publisher(Twist, '/cmd_vel_2', 1)
        self.cmd_pub_3 = self.create_publisher(Twist, '/cmd_vel_3', 1)
        self.cmd_pub_4 = self.create_publisher(Twist, '/cmd_vel_4', 1)
        self.cmd_pub_5 = self.create_publisher(Twist, '/cmd_vel_5', 1)  # 新增
        
        # 控制定时器 (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("🚀 CBF五车安全控制器已启动（带桌边安全约束）")
        self.get_logger().info(f"🎯 安全半径: {self.safety_radius}m, 置信水平: {self.confidence}")
        self.get_logger().info(f"📏 桌面边界: X=[{self.table_boundaries['x_min']:.1f}, {self.table_boundaries['x_max']:.1f}]m, "
                              f"Y=[{self.table_boundaries['y_min']:.1f}, {self.table_boundaries['y_max']:.1f}]m")
        self.get_logger().info(f"🛡️ 边界安全距离: {self.boundary_safety_margin}m")
    
    def record_trajectory_data(self):
        """
        记录各小车轨迹数据到CSV文件
        按照设定的时间间隔记录，避免数据过于密集
        """
        if not self.data_recording_enabled:
            return
        
        # ===== 新增：到达目标后停止记录 =====
        if self.has_reached_target:
            return
            
        # 获取当前时间
        current_time = self.get_clock().now().nanoseconds / 1e9
    
    # 首次记录时设置开始时间（以此为0点）
        if self.recording_start_time is None:
          self.recording_start_time = current_time
          self.last_record_time = 0.0  # 相对时间的起始点
    
       # 计算相对时间（从开始运动到现在的秒数）
        relative_time = current_time - self.recording_start_time
        
        # 检查是否需要记录（按时间间隔采样）
        if relative_time - self.last_record_time < self.data_record_interval:
            return
            
        # 检查所有小车数据是否有效
        if (self.current_1_x is None or self.current_1_y is None or
            self.current_2_x is None or self.current_2_y is None or
            self.current_3_x is None or self.current_3_y is None or
            self.current_4_x is None or self.current_4_y is None or
            self.current_5_x is None or self.current_5_y is None):
            return
        
        # 确定系统状态
        if self.has_reached_target:
            state = "已完成"
        elif self.targetset:
            state = "行进中"
        else:
            state = "等待指令"
        
        # 准备要写入的数据行
        data_row = [
            f"{relative_time:.3f}",           # 时间戳
            f"{self.current_1_x:.4f}",        # 小车1 x
            f"{self.current_1_y:.4f}",        # 小车1 y
            f"{self.current_2_x:.4f}",        # 小车2 x
            f"{self.current_2_y:.4f}",        # 小车2 y
            f"{self.current_3_x:.4f}",        # 小车3 x
            f"{self.current_3_y:.4f}",        # 小车3 y
            f"{self.current_4_x:.4f}",        # 小车4 x
            f"{self.current_4_y:.4f}",        # 小车4 y
            f"{self.current_5_x:.4f}",        # 小车5 x
            f"{self.current_5_y:.4f}",        # 小车5 y
            str(self.targetcar) if self.targetcar else "",  # 目标小车
            f"{self.target_x:.4f}" if self.target_x else "", # 目标x
            f"{self.target_y:.4f}" if self.target_y else "", # 目标y
            state                              # 系统状态
        ]
        
        try:
            self.csv_writer.writerow(data_row)
            self.csv_file.flush()  # 立即写入磁盘，防止数据丢失
            self.last_record_time = relative_time
        except Exception as e:
            self.get_logger().error(f"❌ 数据记录失败: {e}")
    
    def save_data_summary(self):
        """
        保存数据摘要信息，包括运行统计
        """
        if not self.data_recording_enabled:
            return
            
        try:
            summary_filename = self.csv_filename.replace('.csv', '_summary.txt')
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("小车轨迹数据记录摘要\n")
                f.write(f"记录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("【系统参数】\n")
                f.write(f"安全半径: {self.safety_radius}m\n")
                f.write(f"最大速度: {self.max_speed_xy}m/s\n")
                f.write(f"桌面边界: X=[{self.table_boundaries['x_min']:.2f}, {self.table_boundaries['x_max']:.2f}]m, "
                       f"Y=[{self.table_boundaries['y_min']:.2f}, {self.table_boundaries['y_max']:.2f}]m\n\n")
                
                f.write("【目标位置】\n")
                targets = [
                    ("小车1", self.target_1_x, self.target_1_y),
                    ("小车2", self.target_2_x, self.target_2_y),
                    ("小车3", self.target_3_x, self.target_3_y),
                    ("小车4", self.target_4_x, self.target_4_y),
                    ("小车5", self.target_5_x, self.target_5_y)
                ]
                for name, tx, ty in targets:
                    if tx is not None and ty is not None:
                        f.write(f"{name}: ({tx:.4f}, {ty:.4f})\n")
                
                f.write(f"\n选中目标小车: {self.targetcar}\n")
                f.write(f"手势目标位置: ({self.target_x:.4f}, {self.target_y:.4f})\n\n")
                
                f.write("【运行状态】\n")
                f.write(f"所有小车到达目标: {self.has_reached_target}\n")
                f.write(f"边界违规次数: {self.boundary_violation_count}\n")
                
            self.get_logger().info(f"📄 数据摘要已保存至: {summary_filename}")
            
        except Exception as e:
            self.get_logger().error(f"❌ 保存数据摘要失败: {e}")
    
    def pose_callback_1(self, msg):
        """小车1位置回调"""
        self.current_1_x = -msg.x  # 相机到世界坐标系
        self.current_1_y = -msg.y
        self.current_1_z = msg.theta
        
        if self.state_1 == "等待二维码":
            self.state_1 = "前进中"
            self.get_logger().info(f"🎯 收到小车1位姿: X={self.current_1_x:.3f}m, Y={self.current_1_y:.3f}m")
    
    def pose_callback_2(self, msg):
        """小车2位置回调"""
        self.current_2_x = -msg.x  # 相机到世界坐标系
        self.current_2_y = -msg.y  
        self.current_2_z = msg.theta
        
        if self.state_2 == "等待二维码":
            self.state_2 = "前进中"
            self.get_logger().info(f"🎯 收到小车2位姿: X={self.current_2_x:.3f}m, Y={self.current_2_y:.3f}m")
    
    def pose_callback_3(self, msg):
        """小车3位置回调"""
        self.current_3_x = -msg.x  # 相机到世界坐标系
        self.current_3_y = -msg.y  
        self.current_3_z = msg.theta
        
        if self.state_3 == "等待二维码":
            self.state_3 = "前进中"
            self.get_logger().info(f"🎯 收到小车3位姿: X={self.current_3_x:.3f}m, Y={self.current_3_y:.3f}m")
    
    def pose_callback_4(self, msg):
        """小车4位置回调"""
        self.current_4_x = -msg.x  # 相机到世界坐标系
        self.current_4_y = -msg.y  
        self.current_4_z = msg.theta
        
        if self.state_4 == "等待二维码":
            self.state_4 = "前进中"
            self.get_logger().info(f"🎯 收到小车4位姿: X={self.current_4_x:.3f}m, Y={self.current_4_y:.3f}m")
    
    def pose_callback_5(self, msg):
        """小车5位置回调（新增）"""
        self.current_5_x = -msg.x  # 相机到世界坐标系
        self.current_5_y = -msg.y  
        self.current_5_z = msg.theta
        
        if self.state_5 == "等待二维码":
            self.state_5 = "前进中"
            self.get_logger().info(f"🎯 收到小车5位姿: X={self.current_5_x:.3f}m, Y={self.current_5_y:.3f}m")

    

    

    def pose_callback_6(self, msg):
        """手势数据回调 - 选择目标小车并处理安全距离（改进版）

        新逻辑：
        1. 计算所有小车到手势射线的距离和角度
        2. 筛选角度符合要求（<90°）的小车
        3. 如果最小距离和第二小距离比值 < 1.5：
           - 手抬高（Z≤610mm）→ 选择离手势起点更远的车
           - 手放低（Z>610mm）→ 选择离手势起点更近的车
        4. 如果比值 ≥ 1.5：直接选最近的车
        5. 如果只有一辆车符合：直接选它

        这样即使直线上只有一辆车，无论手高还是手低都能选到它。
        """

        # 坐标转换
        self.target_x = (msg.linear.x - self.offset_x) * 0.01
        self.target_y = (msg.linear.y - self.offset_y) * 0.01
        self.trigger_x = (msg.angular.x - self.offset_x) * 0.01
        self.trigger_y = (msg.angular.y - self.offset_y) * 0.01
        self.trigger_z = msg.angular.z

        # 手势起点（桌边中心，固定位置，单位：米）
        # 根据实际场景调整，这里假设老人坐在桌面下方中心
        GESTURE_START_X = 0.0      # 桌边中心X
        GESTURE_START_Y = -0.37    # 桌边中心Y（桌面下边界）

        if (self.targetset == False and 
            self.current_1_x is not None and self.current_1_y is not None and self.current_1_z is not None and
            self.current_2_x is not None and self.current_2_y is not None and self.current_2_z is not None and
            self.current_3_x is not None and self.current_3_y is not None and self.current_3_z is not None and
            self.current_4_x is not None and self.current_4_y is not None and self.current_4_z is not None and
            self.current_5_x is not None and self.current_5_y is not None and self.current_5_z is not None):

            # 计算点到直线的距离
            def point_to_line_distance(px, py, x1, y1, x2, y2):
                """计算点(px, py)到直线(x1,y1)-(x2,y2)的距离"""
                if abs(x2 - x1) < 1e-6 and abs(y2 - y1) < 1e-6:
                    return math.sqrt((px - x1)**2 + (py - y1)**2)

                numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
                denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

                return numerator / denominator

            # 计算点到手势起点的距离（用于判断远近）
            def distance_to_gesture_start(px, py):
                """计算小车到手势起点的欧氏距离"""
                return math.sqrt((px - GESTURE_START_X)**2 + (py - GESTURE_START_Y)**2)

            # 直线端点
            x1, y1 = self.trigger_x, self.trigger_y  # 触发点
            x2, y2 = self.target_x, self.target_y    # 预测点

            # 计算指向向量（触发点 -> 预测点）
            direction_vector = np.array([x2 - x1, y2 - y1])
            direction_norm = np.linalg.norm(direction_vector)

            if direction_norm < 1e-6:
                self.get_logger().warning("⚠ 触发点和预测点重合，无法计算方向")
                return

            # 归一化指向向量
            direction_norm_vector = direction_vector / direction_norm

            # 获取所有小车位置
            car_positions = [
                (1, self.current_1_x, self.current_1_y),
                (2, self.current_2_x, self.current_2_y),
                (3, self.current_3_x, self.current_3_y),
                (4, self.current_4_x, self.current_4_y),
                (5, self.current_5_x, self.current_5_y)
            ]

            # 计算各小车到手势射线的距离、角度和到手势起点的距离
            car_data = []
            for car_id, car_x, car_y in car_positions:
                # 到射线的垂直距离
                distance = point_to_line_distance(car_x, car_y, x1, y1, x2, y2)

                # 计算从触发点到小车的向量夹角
                car_vector = np.array([car_x - x1, car_y - y1])
                car_norm = np.linalg.norm(car_vector)

                if car_norm < 1e-6:
                    angle = 0.0
                    angle_deg = 0.0
                else:
                    car_norm_vector = car_vector / car_norm
                    cos_angle = np.dot(direction_norm_vector, car_norm_vector)
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    angle = math.acos(cos_angle)
                    angle_deg = math.degrees(angle)

                # 到手势起点的距离（判断远近）
                dist_to_start = distance_to_gesture_start(car_x, car_y)

                car_data.append({
                    'id': car_id,
                    'x': car_x,
                    'y': car_y,
                    'distance': distance,            # 到射线的垂直距离
                    'dist_to_start': dist_to_start,  # 到手势起点的距离
                    'angle': angle,
                    'angle_deg': angle_deg
                })

            # 记录所有小车信息
            self.get_logger().info(f"🎯 各小车分析:")
            for car in car_data:
                self.get_logger().info(f"  小车{car['id']}: 射线距离={car['distance']:.3f}m, "
                                      f"起点距离={car['dist_to_start']:.3f}m, 角度={car['angle_deg']:.1f}°")

            # 定义最大允许角度（90度）
            MAX_ANGLE_THRESHOLD = math.pi / 2

            # 筛选角度符合要求的小车（在射线前方半平面内）
            valid_cars = [car for car in car_data if car['angle'] < MAX_ANGLE_THRESHOLD]

            if not valid_cars:
                self.get_logger().warning("⚠ 没有角度符合要求的小车！")
                return

            # 按射线距离从小到大排序
            valid_cars_sorted = sorted(valid_cars, key=lambda x: x['distance'])

            # ==================== 新的目标选择逻辑 ====================
            DISTANCE_RATIO_THRESHOLD = 6  # 距离比阈值，小于此值认为两车距离接近
            HAND_HEIGHT_THRESHOLD = 610     # mm，手高阈值

            selected_car = None
            selection_reason = ""

            if len(valid_cars_sorted) >= 2:
                # 至少有两辆候选车
                min_dist = valid_cars_sorted[0]['distance']
                second_min_dist = valid_cars_sorted[1]['distance']

                # 计算距离比值（防止除零）
                if min_dist < 1e-6:
                    distance_ratio = float('inf')
                else:
                    distance_ratio = second_min_dist / min_dist

                self.get_logger().info(f"📊 距离分析: 最近={min_dist:.3f}m, 第二近={second_min_dist:.3f}m, "
                                      f"比值={distance_ratio:.2f} (阈值={DISTANCE_RATIO_THRESHOLD})")

                #if distance_ratio < DISTANCE_RATIO_THRESHOLD:
                if second_min_dist < 0.1: #0.25
                    # ===== 两车距离接近，需要用手势高度区分远近 =====
                    car_near = valid_cars_sorted[0]   # 射线距离最近
                    car_second = valid_cars_sorted[1]  # 射线距离第二近

                    is_hand_high = self.trigger_z <= HAND_HEIGHT_THRESHOLD

                    self.get_logger().info(f"🤚 手势高度分析: Z={self.trigger_z:.0f}mm, "
                                          f"阈值={HAND_HEIGHT_THRESHOLD}mm, "
                                          f"{'手抬高' if is_hand_high else '手放低'}")

                    if is_hand_high:
                        # ===== 手抬高 → 选择离手势起点更远的车 =====
                        if car_near['dist_to_start'] >= car_second['dist_to_start']:
                            selected_car = car_near
                            selection_reason = (
                                f"手抬高({self.trigger_z:.0f}mm≤{HAND_HEIGHT_THRESHOLD}mm)，"
                                f"两车距离接近(比值{distance_ratio:.2f})，"
                                f"选择更远的车{car_near['id']}"
                                f"(起点距离{car_near['dist_to_start']:.3f}m > "
                                f"{car_second['dist_to_start']:.3f}m)"
                            )
                        else:
                            selected_car = car_second
                            selection_reason = (
                                f"手抬高({self.trigger_z:.0f}mm≤{HAND_HEIGHT_THRESHOLD}mm)，"
                                f"两车距离接近(比值{distance_ratio:.2f})，"
                                f"选择更远的车{car_second['id']}"
                                f"(起点距离{car_second['dist_to_start']:.3f}m > "
                                f"{car_near['dist_to_start']:.3f}m)"
                            )
                    else:
                        # ===== 手放低 → 选择离手势起点更近的车 =====
                        if car_near['dist_to_start'] <= car_second['dist_to_start']:
                            selected_car = car_near
                            selection_reason = (
                                f"手放低({self.trigger_z:.0f}mm>{HAND_HEIGHT_THRESHOLD}mm)，"
                                f"两车距离接近(比值{distance_ratio:.2f})，"
                                f"选择更近的车{car_near['id']}"
                                f"(起点距离{car_near['dist_to_start']:.3f}m < "
                                f"{car_second['dist_to_start']:.3f}m)"
                            )
                        else:
                            selected_car = car_second
                            selection_reason = (
                                f"手放低({self.trigger_z:.0f}mm>{HAND_HEIGHT_THRESHOLD}mm)，"
                                f"两车距离接近(比值{distance_ratio:.2f})，"
                                f"选择更近的车{car_second['id']}"
                                f"(起点距离{car_second['dist_to_start']:.3f}m < "
                                f"{car_near['dist_to_start']:.3f}m)"
                            )
                else:
                    # ===== 两车距离差别大，直接选射线距离最近的 =====
                    selected_car = valid_cars_sorted[0]
                    selection_reason = (
                        f"距离差别大(比值{distance_ratio:.2f}≥{DISTANCE_RATIO_THRESHOLD})，"
                        f"直接选择射线最近的车{selected_car['id']}"
                        f"(距离{selected_car['distance']:.3f}m)"
                    )
            else:
                # ===== 只有一辆符合角度要求的车 =====
                selected_car = valid_cars_sorted[0]
                selection_reason = (
                    f"仅一辆符合角度要求，直接选择车{selected_car['id']}"
                    f"(射线距离{selected_car['distance']:.3f}m, "
                    f"起点距离{selected_car['dist_to_start']:.3f}m)"
                )

            if selected_car is None:
                self.get_logger().warning("⚠ 没有可选的目标小车！")
                return

            self.targetcar = selected_car['id']

            self.get_logger().info(f"🎯 {selection_reason}")
            self.get_logger().info(f"🎯 最终选择小车{self.targetcar} "
                                  f"(射线距离: {selected_car['distance']:.3f}m, "
                                  f"起点距离: {selected_car['dist_to_start']:.3f}m)")
            self.get_logger().info(f"  触发点: ({x1:.3f}, {y1:.3f})")
            self.get_logger().info(f"  预测点: ({x2:.3f}, {y2:.3f})")
            self.get_logger().info(f"  手高度: {self.trigger_z:.0f}mm")

            # 检查预测落点与各小车的距离
            SAFETY_THRESHOLD = 0.2  # 安全距离阈值 30cm 0.25
            SAFETY_BUFFER = 0.2#0.1

            target_to_car_distances = []
            for i in range(1, 6):
                if i != self.targetcar:
                    car_x = getattr(self, f'current_{i}_x')
                    car_y = getattr(self, f'current_{i}_y')
                    distance = math.sqrt((car_x - x2)**2 + (car_y - y2)**2)
                    target_to_car_distances.append((i, distance))

            too_close_cars = []
            for car_id, distance in target_to_car_distances:
                if distance < SAFETY_THRESHOLD:
                    too_close_cars.append((car_id, distance))
                    self.get_logger().warning(f"⚠ 小车{car_id}离预测点太近: {distance:.3f}m")

            # 根据选择的小车设置目标位置
            if self.targetcar == 1:
                self.target_1_x = self.target_x
                self.target_1_y = self.target_y
                self.targetset = True

                for i in range(2, 6):
                    needs_to_move_away = any(car_id == i for car_id, _ in too_close_cars)
                    if needs_to_move_away:
                        car_x = getattr(self, f'current_{i}_x')
                        car_y = getattr(self, f'current_{i}_y')
                        dx = car_x - x2
                        dy = car_y - y2
                        dist = math.sqrt(dx**2 + dy**2)
                        if dist < 1e-6:
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = math.cos(angle)
                            dy = math.sin(angle)
                            dist = 1.0
                        dx_norm = dx / dist
                        dy_norm = dy / dist
                        move_distance = SAFETY_BUFFER
                        new_x = car_x + dx_norm * move_distance
                        new_y = car_y + dy_norm * move_distance
                        new_x = max(self.table_boundaries['x_min'] + 0.1, 
                                   min(self.table_boundaries['x_max'] - 0.1, new_x))
                        new_y = max(self.table_boundaries['y_min'] + 0.1,
                                   min(self.table_boundaries['y_max'] - 0.1, new_y))
                        setattr(self, f'target_{i}_x', new_x)
                        setattr(self, f'target_{i}_y', new_y)
                        self.get_logger().info(f"  设置小车{i}远离到: ({new_x:.3f}, {new_y:.3f})")
                    else:
                        setattr(self, f'target_{i}_x', getattr(self, f'current_{i}_x'))
                        setattr(self, f'target_{i}_y', getattr(self, f'current_{i}_y'))

                self.get_logger().info("✅ 目标位置有效，设置小车1前往吃饭位置")

            elif self.targetcar == 2:
                self.target_2_x = self.target_x
                self.target_2_y = self.target_y
                self.targetset = True

                for i in [1, 3, 4, 5]:
                    if i != 2:
                        needs_to_move_away = any(car_id == i for car_id, _ in too_close_cars)
                        if needs_to_move_away:
                            car_x = getattr(self, f'current_{i}_x')
                            car_y = getattr(self, f'current_{i}_y')
                            dx = car_x - x2
                            dy = car_y - y2
                            dist = math.sqrt(dx**2 + dy**2)
                            if dist < 1e-6:
                                angle = np.random.uniform(0, 2*np.pi)
                                dx = math.cos(angle)
                                dy = math.sin(angle)
                                dist = 1.0
                            dx_norm = dx / dist
                            dy_norm = dy / dist
                            move_distance = SAFETY_BUFFER
                            new_x = car_x + dx_norm * move_distance
                            new_y = car_y + dy_norm * move_distance
                            new_x = max(self.table_boundaries['x_min'] + 0.1, 
                                       min(self.table_boundaries['x_max'] - 0.1, new_x))
                            new_y = max(self.table_boundaries['y_min'] + 0.1,
                                       min(self.table_boundaries['y_max'] - 0.1, new_y))
                            setattr(self, f'target_{i}_x', new_x)
                            setattr(self, f'target_{i}_y', new_y)
                            self.get_logger().info(f"  设置小车{i}远离到: ({new_x:.3f}, {new_y:.3f})")
                        else:
                            setattr(self, f'target_{i}_x', getattr(self, f'current_{i}_x'))
                            setattr(self, f'target_{i}_y', getattr(self, f'current_{i}_y'))

                self.get_logger().info("✅ 目标位置有效，设置小车2前往吃饭位置")

            elif self.targetcar == 3:
                self.target_3_x = self.target_x
                self.target_3_y = self.target_y
                self.targetset = True

                for i in [1, 2, 4, 5]:
                    if i != 3:
                        needs_to_move_away = any(car_id == i for car_id, _ in too_close_cars)
                        if needs_to_move_away:
                            car_x = getattr(self, f'current_{i}_x')
                            car_y = getattr(self, f'current_{i}_y')
                            dx = car_x - x2
                            dy = car_y - y2
                            dist = math.sqrt(dx**2 + dy**2)
                            if dist < 1e-6:
                                angle = np.random.uniform(0, 2*np.pi)
                                dx = math.cos(angle)
                                dy = math.sin(angle)
                                dist = 1.0
                            dx_norm = dx / dist
                            dy_norm = dy / dist
                            move_distance = SAFETY_BUFFER
                            new_x = car_x + dx_norm * move_distance
                            new_y = car_y + dy_norm * move_distance
                            new_x = max(self.table_boundaries['x_min'] + 0.1, 
                                       min(self.table_boundaries['x_max'] - 0.1, new_x))
                            new_y = max(self.table_boundaries['y_min'] + 0.1,
                                       min(self.table_boundaries['y_max'] - 0.1, new_y))
                            setattr(self, f'target_{i}_x', new_x)
                            setattr(self, f'target_{i}_y', new_y)
                            self.get_logger().info(f"  设置小车{i}远离到: ({new_x:.3f}, {new_y:.3f})")
                        else:
                            setattr(self, f'target_{i}_x', getattr(self, f'current_{i}_x'))
                            setattr(self, f'target_{i}_y', getattr(self, f'current_{i}_y'))

                self.get_logger().info("✅ 目标位置有效，设置小车3前往吃饭位置")

            elif self.targetcar == 4:
                self.target_4_x = self.target_x
                self.target_4_y = self.target_y
                self.targetset = True

                for i in [1, 2, 3, 5]:
                    if i != 4:
                        needs_to_move_away = any(car_id == i for car_id, _ in too_close_cars)
                        if needs_to_move_away:
                            car_x = getattr(self, f'current_{i}_x')
                            car_y = getattr(self, f'current_{i}_y')
                            dx = car_x - x2
                            dy = car_y - y2
                            dist = math.sqrt(dx**2 + dy**2)
                            if dist < 1e-6:
                                angle = np.random.uniform(0, 2*np.pi)
                                dx = math.cos(angle)
                                dy = math.sin(angle)
                                dist = 1.0
                            dx_norm = dx / dist
                            dy_norm = dy / dist
                            move_distance = SAFETY_BUFFER
                            new_x = car_x + dx_norm * move_distance
                            new_y = car_y + dy_norm * move_distance
                            new_x = max(self.table_boundaries['x_min'] + 0.1, 
                                       min(self.table_boundaries['x_max'] - 0.1, new_x))
                            new_y = max(self.table_boundaries['y_min'] + 0.1,
                                       min(self.table_boundaries['y_max'] - 0.1, new_y))
                            setattr(self, f'target_{i}_x', new_x)
                            setattr(self, f'target_{i}_y', new_y)
                            self.get_logger().info(f"  设置小车{i}远离到: ({new_x:.3f}, {new_y:.3f})")
                        else:
                            setattr(self, f'target_{i}_x', getattr(self, f'current_{i}_x'))
                            setattr(self, f'target_{i}_y', getattr(self, f'current_{i}_y'))

                self.get_logger().info("✅ 目标位置有效，设置小车4前往吃饭位置")

            elif self.targetcar == 5:
                self.target_5_x = self.target_x
                self.target_5_y = self.target_y
                self.targetset = True

                for i in [1, 2, 3, 4]:
                    if i != 5:
                        needs_to_move_away = any(car_id == i for car_id, _ in too_close_cars)
                        if needs_to_move_away:
                            car_x = getattr(self, f'current_{i}_x')
                            car_y = getattr(self, f'current_{i}_y')
                            dx = car_x - x2
                            dy = car_y - y2
                            dist = math.sqrt(dx**2 + dy**2)
                            if dist < 1e-6:
                                angle = np.random.uniform(0, 2*np.pi)
                                dx = math.cos(angle)
                                dy = math.sin(angle)
                                dist = 1.0
                            dx_norm = dx / dist
                            dy_norm = dy / dist
                            move_distance = SAFETY_BUFFER
                            new_x = car_x + dx_norm * move_distance
                            new_y = car_y + dy_norm * move_distance
                            new_x = max(self.table_boundaries['x_min'] + 0.1, 
                                       min(self.table_boundaries['x_max'] - 0.1, new_x))
                            new_y = max(self.table_boundaries['y_min'] + 0.1,
                                       min(self.table_boundaries['y_max'] - 0.1, new_y))
                            setattr(self, f'target_{i}_x', new_x)
                            setattr(self, f'target_{i}_y', new_y)
                            self.get_logger().info(f"  设置小车{i}远离到: ({new_x:.3f}, {new_y:.3f})")
                        else:
                            setattr(self, f'target_{i}_x', getattr(self, f'current_{i}_x'))
                            setattr(self, f'target_{i}_y', getattr(self, f'current_{i}_y'))

                self.get_logger().info("✅ 目标位置有效，设置小车5前往吃饭位置")

    def calculate_boundary_constraints(self, positions, desired_velocities):
        """
        计算桌边边界约束（新增）
        基于CBF方法为每个小车对每个边界生成安全约束
        返回边界约束矩阵A_boundary和约束向量b_boundary
        """
        N = 5  # 五辆小车
        num_boundaries = 4  # 四个边界：左、右、下、上
        num_constraints = N * num_boundaries  # 每个小车对每个边界都可能有约束
        
        A_boundary = np.zeros((num_constraints, 2 * N))
        b_boundary = np.zeros(num_constraints)
        
        count = 0
        for i in range(N):
            x, y = positions[0, i], positions[1, i]
            
            # 计算到每个边界的距离
            dist_left = x - self.table_boundaries['x_min'] - self.boundary_safety_margin
            dist_right = self.table_boundaries['x_max'] - x - self.boundary_safety_margin
            dist_bottom = y - self.table_boundaries['y_min'] - self.boundary_safety_margin
            dist_top = self.table_boundaries['y_max'] - y - self.boundary_safety_margin
            
            # 左边界约束：防止小车向左跌落
            if dist_left < self.safety_radius_boundary_start:
                A_boundary[count, 2*i] = -1  # -v_x
                A_boundary[count, 2*i] = -1  # -v_x 项
                h_left = dist_left**2 - self.safety_radius_boundary**2
                b_boundary[count] = self.boundary_gain * max(0, h_left)**3
                count += 1
            
            # 右边界约束：防止小车向右跌落
            if dist_right < self.safety_radius_boundary_start:
                A_boundary[count, 2*i] = 1   # v_x 项
                h_right = dist_right**2 - self.safety_radius_boundary**2
                b_boundary[count] = self.boundary_gain * max(0, h_right)**3
                count += 1
            
            # 下边界约束：防止小车向下跌落
            if dist_bottom < self.safety_radius_boundary_start:
                A_boundary[count, 2*i+1] = -1  # -v_y 项
                h_bottom = dist_bottom**2 - self.safety_radius_boundary**2
                b_boundary[count] = self.boundary_gain * max(0, h_bottom)**3
                count += 1
            
            # 上边界约束：防止小车向上跌落
            if dist_top < self.safety_radius_boundary_start:
                A_boundary[count, 2*i+1] = 1   # v_y 项
                h_top = dist_top**2 - self.safety_radius_boundary**2
                b_boundary[count] = self.boundary_gain * max(0, h_top)**3
                count += 1
        
        # 只返回有效的约束
        return A_boundary[:count], b_boundary[:count]
    
    def check_boundary_violation(self, positions):
        """
        检查边界违规（新增）
        返回是否有小车超出安全边界
        """
        violations = 0
        for i in range(5):  # 五辆小车
            x, y = positions[0, i], positions[1, i]
            
            # 检查是否超出安全边界
            if (x < self.table_boundaries['x_min'] + self.boundary_safety_margin or
                x > self.table_boundaries['x_max'] - self.boundary_safety_margin or
                y < self.table_boundaries['y_min'] + self.boundary_safety_margin or
                y > self.table_boundaries['y_max'] - self.boundary_safety_margin):
                violations += 1
        
        return violations > 0
    
    def trap_cdf_inv(self, a, c, delta, sigma):
        """
        梯形分布CDF反函数（MATLAB代码的Python移植）
        输入: a, c - 两个均匀分布的范围 [-a,a] 和 [-c,c]
              delta - 观测的相对位置 x_i - x_j
              sigma - 置信水平 (>0.5)
        输出: b2, b1 - 满足置信水平的边界
        """
        # 确定A（大范围）和C（小范围）
        if a > c:
            A, C = a, c
        else:
            A, C = c, a
        
        # 无不确定性情况
        if A == 0 and C == 0:
            return delta, delta, sigma
        
        # 梯形参数
        h = 1.0 / (2.0 * A)
        
        # 计算面积
        left_triangle_area = 0.5 * 2 * C * h
        rectangle_area = 2 * (A - C) * h
        right_triangle_area = left_triangle_area
        
        area_seq = [left_triangle_area, rectangle_area, right_triangle_area]
        area_vec = [area_seq[0], area_seq[0] + area_seq[1]]
        
        # 处理三角形分布特殊情况
        if abs(A - C) < 1e-5:
            # 三角形分布
            b1 = (A + C) - 2 * C * math.sqrt((1 - sigma) / (1 - area_vec[1]))
            b2 = -b1
        else:
            # 梯形分布
            if sigma > area_vec[1]:
                # 右侧三角形区域
                b1 = (A + C) - 2 * C * math.sqrt((1 - sigma) / (1 - area_vec[1]))
                b2 = -(A + C) + 2 * C * math.sqrt((1 - sigma) / (1 - area_vec[1]))
            elif (sigma > area_vec[0]) and (sigma <= area_vec[1]):
                # 中间矩形区域
                b1 = -(A - C) + (sigma - area_vec[0]) / h
                b2 = -b1
            elif sigma <= area_vec[0]:
                # 左侧三角形区域
                b1 = -(A + C) + 2 * C * math.sqrt(sigma / area_vec[0])
                b2 = -b1
            else:
                raise ValueError("Invalid sigma value")
        
        # 应用偏移
        b1 += delta
        b2 += delta
        
        return b2, b1, sigma
    
    def select_conservative_boundary(self, b1, b2, axis='x'):
        """
        选择保守边界值（基于MATLAB代码逻辑）
        原则：选择最危险（最接近碰撞）的情况
        """
        # 情况1: 边界跨越0点（最危险）
        if (b2 < 0 and b1 > 0) or (b2 > 0 and b1 < 0):
            #self.get_logger().warn(f'机器人{axis}方向距离小于误差边界！')
            return 0.0
        
        # 情况2: 都是负数（机器人i在j左边）
        elif b1 < 0 and b2 < 0:
            # 选择更大的负数（更靠近0，更危险）
            return max(b1, b2)
        
        # 情况3: 都是正数（机器人i在j右边）
        elif b1 > 0 and b2 > 0:
            # 选择更小的正数（更靠近0，更危险）
            return min(b1, b2)
        
        # 情况4: 其他（通常b1≈b2）
        else:
            #self.get_logger().warn(f'机器人{axis}方向无不确定性或sigma=0.5')
            return b1
    
    def calculate_safe_velocities(self, desired_velocities, positions):
        """
        完整的CBF屏障证书 - 计算安全速度（支持五车和边界避障）
        输入: desired_velocities - 2×N期望速度 [vx_world; vy_world]
              positions - 2×N当前位置 [x; y]  
        输出: safe_velocities - 2×N安全速度 [vx_world_safe; vy_world_safe]
        """
        N = 5  # 五辆小车
        
        if N < 2:
            return desired_velocities
        
        # 初始化小车间约束（五车有10个约束：C(5,2)=10）
        num_robot_constraints = N * (N - 1) // 2
        A_robot = np.zeros((num_robot_constraints, 2 * N))
        b_robot = np.zeros(num_robot_constraints)
        
        count = 0
        # 为每对机器人构建约束
        for i in range(N-1):
            for j in range(i+1, N):
                # 提取机器人位置
                x_i, y_i = positions[0, i], positions[1, i]
                x_j, y_j = positions[0, j], positions[1, j]
                
                # 计算不确定性边界
                max_dvij_x = 2 * self.u_rand_span
                max_dvij_y = 2 * self.u_rand_span
                max_dxij_x = abs(x_i - x_j) + 2 * self.x_rand_span
                max_dxij_y = abs(y_i - y_j) + 2 * self.x_rand_span
                
                # 概率边界计算
                b2_x, b1_x, _ = self.trap_cdf_inv(
                    self.x_rand_span, self.x_rand_span, x_i - x_j, self.confidence
                )
                b2_y, b1_y, _ = self.trap_cdf_inv(
                    self.x_rand_span, self.x_rand_span, y_i - y_j, self.confidence
                )
                
                # 选择保守边界
                b_x = self.select_conservative_boundary(b1_x, b2_x, 'x')
                b_y = self.select_conservative_boundary(b1_y, b2_y, 'y')
                
                # 构建约束矩阵
                A_robot[count, (2*i):(2*i+2)] = -2 * np.array([b_x, b_y])
                A_robot[count, (2*j):(2*j+2)] = 2 * np.array([b_x, b_y])
                
                # 计算屏障函数h
                h1 = b_x**2 - self.safety_radius**2 - 2 * max_dvij_x * max_dxij_x / self.barrier_gain
                h2 = b_y**2 - self.safety_radius**2 - 2 * max_dvij_y * max_dxij_y / self.barrier_gain
                h = h1 + h2
                
                b_robot[count] = self.barrier_gain * (h**3)
                count += 1
        
        # 计算边界约束（新增）
        A_boundary, b_boundary = self.calculate_boundary_constraints(positions, desired_velocities)
        
        # 合并所有约束
        if A_boundary.shape[0] > 0:
            A_total = np.vstack([A_robot, A_boundary])
            b_total = np.concatenate([b_robot, b_boundary])
        else:
            A_total = A_robot
            b_total = b_robot
        
        # 输入约束
        A_ineq = np.vstack([np.eye(2*N), -np.eye(2*N)])
        b_ineq = 50000 * np.ones(4*N)
        
        # 合并所有约束
        A_total = np.vstack([A_total, A_ineq])
        b_total = np.concatenate([b_total, b_ineq])
        
        # ==================== QP求解 ====================
        v_desired = desired_velocities.flatten('F')
        H = 2 * np.eye(2*N)
        f = -2 * v_desired
        
        try:
            res = minimize(
                lambda v: 0.5 * v.dot(H).dot(v) + f.dot(v),
                v_desired,
                constraints={'type': 'ineq', 'fun': lambda v: b_total - A_total.dot(v)},
                method='SLSQP',
                options={'maxiter': 50, 'ftol': 1e-6}
            )
            
            if res.success:
                safe_velocities = res.x.reshape(2, N, order='F')
                
                # 记录性能指标
                execution_time = res.get('execution_time', 0)
                if execution_time > 0.04:  # 接近20Hz周期
                    self.get_logger().warn(f'QP求解时间较长: {execution_time:.3f}s')
                
                return safe_velocities
            else:
                self.get_logger().warn("QP求解失败，使用原始速度")
                return desired_velocities
                
        except Exception as e:
            self.get_logger().error(f"QP求解异常: {e}，使用原始速度")
            return desired_velocities
    
    def control_loop(self):
        """主控制循环 - 集成完整的CBF安全保证（五车版本 + 边界避障）"""
        # 检查数据有效性
        if (self.current_1_x is None or self.current_1_y is None or self.current_1_z is None or
            self.current_2_x is None or self.current_2_y is None or self.current_2_z is None or
            self.current_3_x is None or self.current_3_y is None or self.current_3_z is None or
            self.current_4_x is None or self.current_4_y is None or self.current_4_z is None or
            self.current_5_x is None or self.current_5_y is None or self.current_5_z is None or
            self.state_1 == "等待二维码" or self.state_2 == "等待二维码" or 
            self.state_3 == "等待二维码" or self.state_4 == "等待二维码" or
            self.state_5 == "等待二维码" or
            abs(self.current_1_x) > 100 or abs(self.current_2_x) > 100 or 
            abs(self.current_3_x) > 100 or abs(self.current_4_x) > 100 or
            abs(self.current_5_x) > 100 or
            self.targetset==False or self.target_1_x is None or self.target_1_y is None or self.target_2_x is None
            or self.target_2_y is None or self.target_3_x is None or self.target_3_y is None or self.target_4_x is None or self.target_4_y is None
            or self.target_5_x is None or self.target_5_y is None):
            return
        
        # ==================== 1. 计算位置误差 ====================
        error_1_x = self.target_1_x - self.current_1_x
        error_1_y = self.target_1_y - self.current_1_y
        error_1_z = self.target_1_z - self.current_1_z
        
        error_2_x = self.target_2_x - self.current_2_x
        error_2_y = self.target_2_y - self.current_2_y  
        error_2_z = self.target_2_z - self.current_2_z
        
        error_3_x = self.target_3_x - self.current_3_x
        error_3_y = self.target_3_y - self.current_3_y  
        error_3_z = self.target_3_z - self.current_3_z
        
        error_4_x = self.target_4_x - self.current_4_x
        error_4_y = self.target_4_y - self.current_4_y  
        error_4_z = self.target_4_z - self.current_4_z
        
        # 新增第五辆小车误差计算
        error_5_x = self.target_5_x - self.current_5_x
        error_5_y = self.target_5_y - self.current_5_y  
        error_5_z = self.target_5_z - self.current_5_z
        
        # ==================== 2. PID控制计算期望速度 ====================
        # 世界坐标系下的期望速度
        desired_v1_x = self.kp_1_x * error_1_x
        desired_v1_y = self.kp_1_y * error_1_y
        desired_v2_x = self.kp_2_x * error_2_x  
        desired_v2_y = self.kp_2_y * error_2_y
        desired_v3_x = self.kp_3_x * error_3_x
        desired_v3_y = self.kp_3_y * error_3_y
        desired_v4_x = self.kp_4_x * error_4_x
        desired_v4_y = self.kp_4_y * error_4_y
        desired_v5_x = self.kp_5_x * error_5_x
        desired_v5_y = self.kp_5_y * error_5_y
        
        # 角速度控制（保持目标角度）
        omega_1 = self.kp_1_z * error_1_z
        omega_2 = self.kp_2_z * error_2_z
        omega_3 = self.kp_3_z * error_3_z
        omega_4 = self.kp_4_z * error_4_z
        omega_5 = self.kp_5_z * error_5_z
        
        # 速度限幅 (x,y平面)
        speed_1_xy = math.sqrt(desired_v1_x**2 + desired_v1_y**2)
        if speed_1_xy > self.max_speed_xy:
            desired_v1_x = desired_v1_x * (self.max_speed_xy / speed_1_xy)
            desired_v1_y = desired_v1_y * (self.max_speed_xy / speed_1_xy)
            
        speed_2_xy = math.sqrt(desired_v2_x**2 + desired_v2_y**2)  
        if speed_2_xy > self.max_speed_xy:
            desired_v2_x = desired_v2_x * (self.max_speed_xy / speed_2_xy)
            desired_v2_y = desired_v2_y * (self.max_speed_xy / speed_2_xy)
            
        speed_3_xy = math.sqrt(desired_v3_x**2 + desired_v3_y**2)  
        if speed_3_xy > self.max_speed_xy:
            desired_v3_x = desired_v3_x * (self.max_speed_xy / speed_3_xy)
            desired_v3_y = desired_v3_y * (self.max_speed_xy / speed_3_xy)
            
        speed_4_xy = math.sqrt(desired_v4_x**2 + desired_v4_y**2)  
        if speed_4_xy > self.max_speed_xy:
            desired_v4_x = desired_v4_x * (self.max_speed_xy / speed_4_xy)
            desired_v4_y = desired_v4_y * (self.max_speed_xy / speed_4_xy)
        
        # 新增第五辆小车速度限幅
        speed_5_xy = math.sqrt(desired_v5_x**2 + desired_v5_y**2)  
        if speed_5_xy > self.max_speed_xy:
            desired_v5_x = desired_v5_x * (self.max_speed_xy / speed_5_xy)
            desired_v5_y = desired_v5_y * (self.max_speed_xy / speed_5_xy)
        
        # 角速度限幅
        omega_1 = max(min(omega_1, self.max_speed_z), -self.max_speed_z)
        omega_2 = max(min(omega_2, self.max_speed_z), -self.max_speed_z)
        omega_3 = max(min(omega_3, self.max_speed_z), -self.max_speed_z)
        omega_4 = max(min(omega_4, self.max_speed_z), -self.max_speed_z)
        omega_5 = max(min(omega_5, self.max_speed_z), -self.max_speed_z)
        
        # ==================== 3. CBF安全修正（包含边界避障） ====================
        # 准备CBF输入数据
        desired_velocities = np.array([[desired_v1_x, desired_v2_x, desired_v3_x, desired_v4_x, desired_v5_x],
                                      [desired_v1_y, desired_v2_y, desired_v3_y, desired_v4_y, desired_v5_y]])
        
        positions = np.array([[self.current_1_x, self.current_2_x, self.current_3_x, self.current_4_x, self.current_5_x],
                             [self.current_1_y, self.current_2_y, self.current_3_y, self.current_4_y, self.current_5_y]])
        
        # 应用CBF屏障证书（包含边界约束）
        safe_velocities = self.calculate_safe_velocities(desired_velocities, positions)
        
        # 提取安全速度
        safe_v1_x, safe_v2_x, safe_v3_x, safe_v4_x, safe_v5_x = safe_velocities[0, 0], safe_velocities[0, 1], safe_velocities[0, 2], safe_velocities[0, 3], safe_velocities[0, 4]
        safe_v1_y, safe_v2_y, safe_v3_y, safe_v4_y, safe_v5_y = safe_velocities[1, 0], safe_velocities[1, 1], safe_velocities[1, 2], safe_velocities[1, 3], safe_velocities[1, 4]
        
        # ==================== 4. 坐标系转换 ====================
        # 世界坐标系 → 车身坐标系
        theta_1 = self.current_1_z
        theta_2 = self.current_2_z
        theta_3 = self.current_3_z
        theta_4 = self.current_4_z
        theta_5 = self.current_5_z
        
        cmd_msg_1 = Twist()
        cmd_msg_2 = Twist()
        cmd_msg_3 = Twist()
        cmd_msg_4 = Twist()
        cmd_msg_5 = Twist()
        
        # 检查是否到达目标
        position_error_1 = math.sqrt(error_1_x**2 + error_1_y**2)
        position_error_2 = math.sqrt(error_2_x**2 + error_2_y**2)
        position_error_3 = math.sqrt(error_3_x**2 + error_3_y**2)
        position_error_4 = math.sqrt(error_4_x**2 + error_4_y**2)
        position_error_5 = math.sqrt(error_5_x**2 + error_5_y**2)
        orientation_error_1 = abs(error_1_z)
        orientation_error_2 = abs(error_2_z)
        orientation_error_3 = abs(error_3_z)
        orientation_error_4 = abs(error_4_z)
        orientation_error_5 = abs(error_5_z)
        
        self.record_trajectory_data()
        
        if ((position_error_1 < self.stop_tolerance_xy and 
            position_error_2 < self.stop_tolerance_xy and
            position_error_3 < self.stop_tolerance_xy and
            position_error_4 < self.stop_tolerance_xy and
            position_error_5 < self.stop_tolerance_xy and
            orientation_error_1 < self.stop_tolerance_z and
            orientation_error_2 < self.stop_tolerance_z and
            orientation_error_3 < self.stop_tolerance_z and
            orientation_error_4 < self.stop_tolerance_z and
            orientation_error_5 < self.stop_tolerance_z) or (self.has_reached_target==True)):
            
            # 到达目标，停止
            cmd_msg_1.linear.x = 0.0
            cmd_msg_1.linear.y = 0.0  
            cmd_msg_1.angular.z = 0.0
            
            cmd_msg_2.linear.x = 0.0
            cmd_msg_2.linear.y = 0.0
            cmd_msg_2.angular.z = 0.0
            
            cmd_msg_3.linear.x = 0.0
            cmd_msg_3.linear.y = 0.0
            cmd_msg_3.angular.z = 0.0
            
            cmd_msg_4.linear.x = 0.0
            cmd_msg_4.linear.y = 0.0
            cmd_msg_4.angular.z = 0.0
            
            cmd_msg_5.linear.x = 0.0
            cmd_msg_5.linear.y = 0.0
            cmd_msg_5.angular.z = 0.0
            
            if not self.has_reached_target:
                self.get_logger().info("🎉 所有小车已到达目标位置！")
                self.has_reached_target = True
        else:
            # 应用安全速度控制
            # 小车1：世界坐标系 → 车身坐标系
            cmd_msg_1.linear.x = safe_v1_x * math.cos(theta_1) + safe_v1_y * math.sin(theta_1)
            cmd_msg_1.linear.y = -safe_v1_x * math.sin(theta_1) + safe_v1_y * math.cos(theta_1)
            cmd_msg_1.angular.z = omega_1
            
            # 小车2：世界坐标系 → 车身坐标系  
            cmd_msg_2.linear.x = safe_v2_x * math.cos(theta_2) + safe_v2_y * math.sin(theta_2)
            cmd_msg_2.linear.y = -safe_v2_x * math.sin(theta_2) + safe_v2_y * math.cos(theta_2)
            cmd_msg_2.angular.z = omega_2
            
            # 小车3：世界坐标系 → 车身坐标系  
            cmd_msg_3.linear.x = safe_v3_x * math.cos(theta_3) + safe_v3_y * math.sin(theta_3)
            cmd_msg_3.linear.y = -safe_v3_x * math.sin(theta_3) + safe_v3_y * math.cos(theta_3)
            cmd_msg_3.angular.z = omega_3
            
            # 小车4：世界坐标系 → 车身坐标系  
            cmd_msg_4.linear.x = safe_v4_x * math.cos(theta_4) + safe_v4_y * math.sin(theta_4)
            cmd_msg_4.linear.y = -safe_v4_x * math.sin(theta_4) + safe_v4_y * math.cos(theta_4)
            cmd_msg_4.angular.z = omega_4
            
            # 小车5：世界坐标系 → 车身坐标系
            cmd_msg_5.linear.x = safe_v5_x * math.cos(theta_5) + safe_v5_y * math.sin(theta_5)
            cmd_msg_5.linear.y = -safe_v5_x * math.sin(theta_5) + safe_v5_y * math.cos(theta_5)
            cmd_msg_5.angular.z = omega_5
        
        # ==================== 5. 安全监控 ====================
        # 计算最小车距
        min_distance_12 = math.sqrt((self.current_1_x - self.current_2_x)**2 + 
                                  (self.current_1_y - self.current_2_y)**2)
        min_distance_13 = math.sqrt((self.current_1_x - self.current_3_x)**2 + 
                                  (self.current_1_y - self.current_3_y)**2)
        min_distance_14 = math.sqrt((self.current_1_x - self.current_4_x)**2 + 
                                  (self.current_1_y - self.current_4_y)**2)
        min_distance_15 = math.sqrt((self.current_1_x - self.current_5_x)**2 + 
                                  (self.current_1_y - self.current_5_y)**2)
        min_distance_23 = math.sqrt((self.current_2_x - self.current_3_x)**2 + 
                                  (self.current_2_y - self.current_3_y)**2)
        min_distance_24 = math.sqrt((self.current_2_x - self.current_4_x)**2 + 
                                  (self.current_2_y - self.current_4_y)**2)
        min_distance_25 = math.sqrt((self.current_2_x - self.current_5_x)**2 + 
                                  (self.current_2_y - self.current_5_y)**2)
        min_distance_34 = math.sqrt((self.current_3_x - self.current_4_x)**2 + 
                                  (self.current_3_y - self.current_4_y)**2)
        min_distance_35 = math.sqrt((self.current_3_x - self.current_5_x)**2 + 
                                  (self.current_3_y - self.current_5_y)**2)
        min_distance_45 = math.sqrt((self.current_4_x - self.current_5_x)**2 + 
                                  (self.current_4_y - self.current_5_y)**2)
        
        min_distance = min(min_distance_12, min_distance_13, min_distance_14, min_distance_15,
                          min_distance_23, min_distance_24, min_distance_25,
                          min_distance_34, min_distance_35, min_distance_45)
        
        # ==================== 6. 记录轨迹数据 ====================
        #self.record_trajectory_data()
        
        # ==================== 7. 发布控制命令 ====================
        # 设置其他速度为0
        cmd_msg_1.linear.z = 0.0
        cmd_msg_1.angular.x = 0.0
        cmd_msg_1.angular.y = 0.0
        
        cmd_msg_2.linear.z = 0.0  
        cmd_msg_2.angular.x = 0.0
        cmd_msg_2.angular.y = 0.0
        
        cmd_msg_3.linear.z = 0.0  
        cmd_msg_3.angular.x = 0.0
        cmd_msg_3.angular.y = 0.0
        
        cmd_msg_4.linear.z = 0.0  
        cmd_msg_4.angular.x = 0.0
        cmd_msg_4.angular.y = 0.0
        
        cmd_msg_5.linear.z = 0.0
        cmd_msg_5.angular.x = 0.0
        cmd_msg_5.angular.y = 0.0
        
        self.cmd_pub_1.publish(cmd_msg_1)
        self.cmd_pub_2.publish(cmd_msg_2)
        self.cmd_pub_3.publish(cmd_msg_3)
        self.cmd_pub_4.publish(cmd_msg_4)
        self.cmd_pub_5.publish(cmd_msg_5)
    
    def stop_cars(self):
        """紧急停止所有小车"""
        cmd_msg = Twist()
        cmd_msg.linear.x = cmd_msg.linear.y = cmd_msg.linear.z = 0.0
        cmd_msg.angular.x = cmd_msg.angular.y = cmd_msg.angular.z = 0.0
        
        self.cmd_pub_1.publish(cmd_msg)
        self.cmd_pub_2.publish(cmd_msg)
        self.cmd_pub_3.publish(cmd_msg)
        self.cmd_pub_4.publish(cmd_msg)
        self.cmd_pub_5.publish(cmd_msg)
        self.get_logger().info("🛑 紧急停止已触发")
    
    def destroy_node(self):
        """节点销毁时停止小车并保存数据"""
        self.stop_cars()
        
        # 保存数据摘要
        if self.data_recording_enabled:
            self.save_data_summary()
            # 关闭CSV文件
            if hasattr(self, 'csv_file') and self.csv_file:
                self.csv_file.close()
                self.get_logger().info(f"💾 轨迹数据已保存至: {self.csv_filename}")
        
        self.get_logger().info("👋 CBF控制器关闭，小车已安全停止")
        super().destroy_node()

def main():
    rclpy.init()
    controller = CBFFiveCarController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("👋 用户中断，安全关闭控制器")
    except Exception as e:
        controller.get_logger().error(f"控制器异常: {e}")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()