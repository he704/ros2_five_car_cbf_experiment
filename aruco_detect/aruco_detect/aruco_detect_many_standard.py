#!/usr/bin/env python3
""" 
ArUco检测节点 - 为每个有效ID发布独立位姿话题
发布话题: /aruco_pose_<id> (例如: /aruco_pose_0, /aruco_pose_1, /aruco_pose_2)
消息类型: geometry_msgs/Pose2D
"""

import numpy as np
import time
import cv2
import cv2.aruco as aruco
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
import sys

# 导入更新后的utils
try:
    from .my_utils import PoseToRTmatrix, Q_RTmatrixToPose
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("警告: 无法导入utils模块，使用简化版本")

from .my_parameter import RGB_MTX_0, RGB_DIST_0, T_base_camera_0

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # 相机内参 realsense
        self.mtx = RGB_MTX_0

        # 相机畸变参数
        self.dist = RGB_DIST_0

        # 相机外参数
        self.T_base_cam = T_base_camera_0

        # 发布者字典，每个ID对应一个发布者
        self.pose_publishers = {}
        self.publish_rate = 30  # Hz

        # 标记参数
        self.marker_size = 0.071  # 标记尺寸（米）
        #self.valid_marker_ids = [0, 1, 2]  # 有效的标记ID列表
        self.valid_marker_ids = [0, 1, 2, 3, 4]

        # 为每个有效ID创建发布者
        for marker_id in self.valid_marker_ids:
            topic_name = f'/aruco_pose_{marker_id}'
            publisher = self.create_publisher(Pose2D, topic_name, 1)
            self.pose_publishers[marker_id] = publisher
            self.get_logger().info(f'创建发布者: {topic_name}')

        # 打开摄像头
        self.cap = cv2.VideoCapture(4)  # 4：realsense RGB
        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头，请检查摄像头ID")
            sys.exit(1)

        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 字典，数值对应的二维码
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters_create()

        # 设置字体类型和字体缩放比例
        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5
        self.color = (0, 255, 0)  # 这里是bgr
        self.thickness = 2

        # 用于记录未检测到的帧数
        self.not_detected_frames = {id: 0 for id in self.valid_marker_ids}
        self.max_not_detected_frames = 10  # 连续10帧未检测到则发布特殊值

        # 角度平滑处理参数
        self.yaw_threshold_deg = 20.0  # 角度跳变阈值（度）
        # 存储每个ID的上次有效位姿（包括位置和旋转）
        self.last_valid_pose = {}
        for marker_id in self.valid_marker_ids:
            self.last_valid_pose[marker_id] = {
                'tvec': None,      # 平移向量 [x, y, z]
                'rvec': None,      # 旋转向量
                'yaw': None,       # yaw角（弧度）
            }

        self.show_image = True
        self.running = True

        self.get_logger().info('ArUco检测节点已启动')
        self.get_logger().info(f'发布频率: {self.publish_rate}Hz')
        self.get_logger().info(f'有效标记ID: {self.valid_marker_ids}')
        self.get_logger().info(f'发布话题: {[f"/aruco_pose_{id}" for id in self.valid_marker_ids]}')
        self.get_logger().info(f'角度跳变阈值: {self.yaw_threshold_deg}degrees')

        # 启动主循环
        self.timer = self.create_timer(1.0/self.publish_rate, self.process_frame)
        self.last_frame_time = time.time()

    def safe_len(self, obj):
        """安全地获取对象的长度，避免TypeError"""
        try:
            return len(obj)
        except TypeError:
            return 0

    def rotation_vector_to_euler(self, rvec):
        """将旋转向量转换为欧拉角"""
        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # 提取绕Z轴的角度（偏航角yaw）
        theta_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        return theta_z

    def angle_diff_deg(self, angle1, angle2):
        """计算两个角度之间的最小差值（度），处理360度环绕"""
        if angle1 is None or angle2 is None:
            return float('inf')
        diff = abs(np.degrees(angle1 - angle2))
        # 处理360度环绕情况（例如 350度和10度的差值是20度，不是340度）
        while diff > 180:
            diff = 360 - diff
        return diff

    def smooth_pose(self, marker_id, raw_tvec, raw_rvec, raw_yaw):
        """
        平滑处理位姿，检测yaw跳变并返回平滑后的位姿

        策略：
        - 如果yaw跳变超过阈值：保留当前tvec（位置），但rvec和yaw沿用上次有效值
        - 如果正常：更新所有值为当前值

        参数:
            marker_id: 标记ID
            raw_tvec: 原始平移向量 [x, y, z]
            raw_rvec: 原始旋转向量
            raw_yaw: 原始yaw角（弧度）

        返回:
            smoothed_tvec: 平滑后的平移向量
            smoothed_rvec: 平滑后的旋转向量
            smoothed_yaw: 平滑后的yaw角
            is_smoothed: 是否使用了平滑（True表示检测到跳变）
        """
        last_pose = self.last_valid_pose.get(marker_id)
        last_yaw = last_pose['yaw'] if last_pose else None

        # 如果是第一次检测到该ID，直接记录并返回
        if last_yaw is None:
            self.last_valid_pose[marker_id] = {
                'tvec': raw_tvec.copy(),
                'rvec': raw_rvec.copy(),
                'yaw': raw_yaw,
            }
            return raw_tvec, raw_rvec, raw_yaw, False

        # 计算角度差值（度）
        diff_deg = self.angle_diff_deg(raw_yaw, last_yaw)

        # 如果跳变超过阈值，保留位置，但旋转沿用上次值
        if diff_deg > self.yaw_threshold_deg:
            self.get_logger().debug(
                f'ID:{marker_id} 检测到角度跳变: {np.degrees(last_yaw):.1f} -> {np.degrees(raw_yaw):.1f} '
                f'(差值: {diff_deg:.1f}degrees > {self.yaw_threshold_deg}degrees)，'
                f'位置更新，旋转沿用上次值',
                throttle_duration_sec=0.5
            )
            # 保留当前tvec（位置），rvec和yaw沿用上次
            smoothed_tvec = raw_tvec.copy()
            smoothed_rvec = last_pose['rvec'].copy()
            smoothed_yaw = last_pose['yaw']

            # 更新存储：位置用新的，旋转用旧的
            self.last_valid_pose[marker_id] = {
                'tvec': smoothed_tvec,
                'rvec': smoothed_rvec,
                'yaw': smoothed_yaw,
            }
            return smoothed_tvec, smoothed_rvec, smoothed_yaw, True
        else:
            # 角度正常，更新所有值
            self.last_valid_pose[marker_id] = {
                'tvec': raw_tvec.copy(),
                'rvec': raw_rvec.copy(),
                'yaw': raw_yaw,
            }
            return raw_tvec, raw_rvec, raw_yaw, False

    def publish_pose_for_id(self, marker_id, x, y, theta_z, detected=True):
        """发布特定ID的位姿信息"""
        if marker_id not in self.pose_publishers:
            # 动态创建发布者（如果检测到未预先定义的ID）
            topic_name = f'/aruco_pose_{marker_id}'
            self.pose_publishers[marker_id] = self.create_publisher(Pose2D, topic_name, 1)
            self.get_logger().warning(f'动态创建发布者: {topic_name}')

        pose_msg = Pose2D()

        if detected:
            pose_msg.x = float(x)  # X坐标（米）
            pose_msg.y = float(y)  # Y坐标（米）
            pose_msg.theta = float(theta_z)  # 绕Z轴角度（弧度）
            self.not_detected_frames[marker_id] = 0
        else:
            # 未检测到标记时发布特殊值
            pose_msg.x = -999.0
            pose_msg.y = -999.0
            pose_msg.theta = 0.0
            self.not_detected_frames[marker_id] += 1
            # 清除上次有效位姿记录，下次检测时重新初始化
            self.last_valid_pose[marker_id] = {'tvec': None, 'rvec': None, 'yaw': None}

        self.pose_publishers[marker_id].publish(pose_msg)

        if detected:
            self.get_logger().info(
                f'发布位姿 - ID:{marker_id} | X:{x:.3f}m, Y:{y:.3f}m, θ:{np.degrees(theta_z):.1f}degrees',
                throttle_duration_sec=0.5)  # 限制日志频率

    def process_frame(self):
        """处理每一帧图像并为每个检测到的标记发布位姿"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("无法读取摄像头帧")
                return

            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测ArUco标记
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters)

            # 记录当前帧检测到的ID
            detected_ids_in_frame = []

            # 用于存储有效标记的corners和ids（仅用于可视化）
            valid_corners = []
            valid_ids = []

            if ids is not None and self.safe_len(ids) > 0:
                # 确保ids是numpy数组并正确处理
                ids = np.array(ids).flatten()
                num_markers = self.safe_len(ids)

                for i in range(num_markers):
                    try:
                        current_id = int(ids[i])

                        # 只处理有效的标记ID
                        if current_id in self.valid_marker_ids:
                            # 估计位姿
                            rvec_obj, tvec_obj, _ = aruco.estimatePoseSingleMarkers(
                                corners[i], self.marker_size, self.mtx, self.dist)

                            if rvec_obj is not None and tvec_obj is not None:
                                rvec_obj = rvec_obj.squeeze()
                                tvec_obj = tvec_obj.squeeze()

                                # 计算原始绕Z轴的角度
                                raw_theta_z = self.rotation_vector_to_euler(rvec_obj)

                                # 平滑处理位姿，检测yaw跳变
                                # 跳变时：保留tvec（位置），沿用上次rvec和yaw
                                smoothed_tvec, smoothed_rvec, smoothed_theta_z, is_smoothed =                                     self.smooth_pose(current_id, tvec_obj, rvec_obj, raw_theta_z)

                                # 记录检测到的ID
                                detected_ids_in_frame.append(current_id)

                                # 收集有效的corners和ids用于可视化
                                valid_corners.append(corners[i])
                                valid_ids.append(current_id)

                                # ==================== 关键修改 ====================
                                # 发布ROS消息：使用原始数据（真实位姿）
                                self.publish_pose_for_id(
                                    current_id, tvec_obj[0], tvec_obj[1], raw_theta_z, True)

                                # 可视化绘图：使用平滑后的数据（视频更美观）
                                # smoothed_tvec, smoothed_rvec, smoothed_theta_z 用于下方绘图代码

                                # 绘制可视化信息
                                if self.show_image:
                                    # 绘制坐标系轴（使用平滑后的旋转向量）
                                    try:
                                         cv2.drawFrameAxes(frame, self.mtx, self.dist, 
                                                          smoothed_rvec, smoothed_tvec, 0.1)
                                    except:
                                         cv2.aruco.drawAxis(frame, self.mtx, self.dist, 
                                                           smoothed_rvec, smoothed_tvec, 0.1)

                                    # 显示位置和角度信息
                                    corner_point = corners[i][0][0].astype(int)

                                    # 如果使用了平滑，在显示中标注
                                    yaw_text = f"Yaw: {np.degrees(smoothed_theta_z):.1f}degrees"
                                    # if is_smoothed:
                                    #     yaw_text += "*"  # 添加标记表示使用了平滑

                                    text_lines = [
                                        f"ID: {current_id}",
                                        f"X: {-smoothed_tvec[0]:.3f}m",
                                        f"Y: {-smoothed_tvec[1]:.3f}m",
                                        yaw_text
                                    ]

                                    # 如果使用了平滑，改变文字颜色提示
                                    text_color =  self.color  # 黄色表示平滑

                                    for j, text in enumerate(text_lines):
                                        y_offset = corner_point[1] - 80 + j * 25
                                        cv2.putText(frame, text, (corner_point[0]-60, y_offset), 
                                                   self.fontFace, self.fontScale, text_color, self.thickness)

                                    # 绘制方向箭头（使用平滑后的角度）
                                    arrow_length = 50
                                    end_x = int(corner_point[0] + arrow_length * np.cos(smoothed_theta_z))
                                    end_y = int(corner_point[1] + arrow_length * np.sin(smoothed_theta_z))
                                    arrow_color = (0, 255, 0)  # 黄色表示平滑
                                    cv2.arrowedLine(frame, tuple(corner_point), (end_x, end_y), 
                                                   arrow_color, 3)

                    except Exception as marker_error:
                        self.get_logger().warn(f"处理标记时出错: {marker_error}")

                # 只绘制有效的标记
                if valid_corners and self.show_image:
                    aruco.drawDetectedMarkers(frame, valid_corners, np.array(valid_ids))

            # 处理未在当前帧中检测到的有效ID
            for marker_id in self.valid_marker_ids:
                if marker_id not in detected_ids_in_frame:
                    # 如果连续多帧未检测到，发布未检测到的特殊值
                    if self.not_detected_frames[marker_id] >= self.max_not_detected_frames:
                        self.publish_pose_for_id(marker_id, 0, 0, 0, False)

            # 如果没有检测到任何 ID，显示提示
            if not detected_ids_in_frame and self.show_image:
                cv2.putText(frame, "No marker detected", (50, 50), 
                           self.fontFace, 1.0, (0, 0, 255), 2)

            # 显示图像
            if self.show_image:
                # 计算FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_frame_time) if (current_time - self.last_frame_time) > 0 else 0
                self.last_frame_time = current_time

                # 显示FPS
                # cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                #            self.fontFace, 0.7, (255, 0, 0), 2)

                # # 显示发布状态
                # cv2.putText(frame, f"Topics: /aruco_pose_<id>", (10, 60), 
                #            self.fontFace, 0.7, (255, 255, 0), 2)

                # # 显示有效ID
                # cv2.putText(frame, f"Valid IDs: {self.valid_marker_ids}", (10, 90), 
                #            self.fontFace, 0.7, (255, 255, 0), 2)

                # # 显示当前检测到的ID
                # if detected_ids_in_frame:
                #     detected_str = f"Detected: {detected_ids_in_frame}"
                # else:
                #     detected_str = "Detected: None"
                # cv2.putText(frame, detected_str, (10, 120), 
                #            self.fontFace, 0.7, (0, 255, 0) if detected_ids_in_frame else (0, 0, 255), 2)

                cv2.imshow('ArUco Detector', frame)

                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.shutdown()
                elif key == ord('s'):
                    filename = f"aruco_detection_{time.time():.0f}.jpg"
                    cv2.imwrite(filename, frame)
                    self.get_logger().info(f"截图已保存: {filename}")

        except Exception as e:
            self.get_logger().error(f"处理帧时出错: {str(e)}")

    def shutdown(self):
        """清理资源并关闭节点"""
        self.get_logger().info("正在关闭节点...")
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    try:
        detector = ArucoDetectorNode()
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info("接收到键盘中断信号")
    except Exception as e:
        detector.get_logger().error(f"节点运行异常: {str(e)}")
    finally:
        if 'detector' in locals():
            detector.shutdown()

    print("ArUco检测节点已退出")

if __name__ == '__main__':
    main()