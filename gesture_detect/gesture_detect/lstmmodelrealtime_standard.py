#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Point, Twist
import pyrealsense2 as rs
import mediapipe as mp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- Config ---------------- #
MODEL_PATH = "/home/he/gesture_ws/src/gesture_detect/gesture_detect/lstmmodel.pt"  # 你的LSTM模型路径
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)
WINDOW_LEN = 30
START_THRESHOLD = 0.3
MIN_MOVING_FRAMES = 10
FPS = 30

# 硬编码的单应性矩阵H
H = np.array([
    [ 1.29404649e-01, -3.51066751e-03, -3.52511784e+01],
    [ 8.05531087e-03, -1.32224425e-01,  7.02940429e+01],
    [ 9.60557291e-05, -1.62798271e-04,  1.00000000e+00]
], dtype=np.float32)

# 归一化常量
X_DEN, Y_DEN, Z_DEN = 70.0, 60.0, 1500.0
VX_DEN, VY_DEN, VZ_DEN = 10.0, 10.0, 500.0
AX_DEN, AY_DEN, AZ_DEN = 10.0, 10.0, 500.0

# 颜色
BLUE = (255, 128, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)

# ---------------- 模型定义 ---------------- #
class LSTMProgressModel(nn.Module):
    def __init__(self, input_dim=9, hidden=96, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.head_pos = nn.Linear(hidden, 3)
        self.head_prog = nn.Linear(hidden, 1)
        self.head_time = nn.Linear(hidden, 1)
        
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        h = hn[-1]
        pos = self.head_pos(h)
        prog = torch.sigmoid(self.head_prog(h))
        t_rem = torch.relu(self.head_time(h))
        return pos, prog, t_rem

# ---------------- 工具函数 ---------------- #
def compute_velocity_acceleration(traj):
    """计算速度和加速度"""
    v = np.diff(traj, axis=0, prepend=traj[0:1])
    a = np.diff(v, axis=0, prepend=v[0:1])
    return np.concatenate([traj, v, a], axis=1)

def to_features(x):
    """特征归一化"""
    x = x.copy()
    x[:, 0] /= X_DEN
    x[:, 1] /= Y_DEN
    x[:, 2] /= Z_DEN
    x[:, 3] /= VX_DEN
    x[:, 4] /= VY_DEN
    x[:, 5] /= VZ_DEN
    x[:, 6] /= AX_DEN
    x[:, 7] /= AY_DEN
    x[:, 8] /= AZ_DEN
    return x

def is_static_xy(p1, p2, thr=START_THRESHOLD):
    """判断手部是否静止"""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    return np.linalg.norm(p1[:2] - p2[:2]) < thr

def px2cm(H, px, py):
    """像素坐标转厘米坐标"""
    if H is None:
        return None
    pt = np.array([[[px, py]]], np.float32)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])

def cm2px(H, x, y):
    """厘米坐标转像素坐标"""
    if H is None:
        return None
    Hinv = np.linalg.inv(H)
    pt = np.array([[[x, y]]], np.float32)
    out = cv2.perspectiveTransform(pt, Hinv)[0, 0]
    return int(round(out[0])), int(round(out[1]))

def xy_err_cm(pred, true):
    """计算XY平面误差"""
    return float(np.linalg.norm(np.asarray(pred)[:2] - np.asarray(true)[:2]))

def z_err_mm(pred, true):
    """计算Z方向误差"""
    return float(abs(pred[2] - true[2]))

def save_result_plot(traj, true_land, pred_info, save_path):
    """保存结果可视化图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    traj_array = np.array(traj)
    
    # 1. XY平面轨迹
    axes[0, 0].plot(traj_array[:, 0], traj_array[:, 1], 'b-', alpha=0.5, label='Trajectory')
    if true_land is not None:
        axes[0, 0].scatter(true_land[0], true_land[1], c='g', s=100, marker='o', label='True landing')
    if pred_info["pred"] is not None:
        axes[0, 0].scatter(pred_info["pred"][0], pred_info["pred"][1], c='r', s=100, marker='^', label='Predicted')
    axes[0, 0].set_xlabel("X (cm)")
    axes[0, 0].set_ylabel("Y (cm)")
    axes[0, 0].set_title("XY Plane")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 深度随时间变化
    if len(traj_array) > 0:
        time_axis = np.arange(len(traj_array)) / FPS
        axes[0, 1].plot(time_axis, traj_array[:, 2], 'b-', label='Depth')
        if pred_info["idx"] is not None and pred_info["idx"] < len(traj_array):
            axes[0, 1].axvline(x=pred_info["idx"]/FPS, color='r', linestyle='--', label=f'Trigger: {pred_info["idx"]/FPS:.2f}s')
        axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Depth (mm)")
    axes[0, 1].set_title("Depth vs Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].invert_yaxis()  # 深度越小越靠近相机
    
    # 3. 3D轨迹
    ax3d = fig.add_subplot(2, 2, (3, 4), projection='3d')
    ax3d.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 'b-', alpha=0.5)
    if true_land is not None:
        ax3d.scatter(true_land[0], true_land[1], true_land[2], c='g', s=100, marker='o')
    if pred_info["pred"] is not None:
        ax3d.scatter(pred_info["pred"][0], pred_info["pred"][1], pred_info["pred"][2], c='r', s=100, marker='^')
    ax3d.set_xlabel("X (cm)")
    ax3d.set_ylabel("Y (cm)")
    ax3d.set_zlabel("Depth (mm)")
    ax3d.set_title("3D Trajectory")
    ax3d.invert_zaxis()  # 深度越小越靠近相机
    
    # 4. 文本信息
    info_text = f"Experiment Results\n\n"
    info_text += f"Total frames: {len(traj)}\n"
    info_text += f"Total time: {len(traj)/FPS:.2f}s\n\n"
    
    if true_land is not None:
        info_text += f"True landing:\n"
        info_text += f"  X: {true_land[0]:.1f}cm\n"
        info_text += f"  Y: {true_land[1]:.1f}cm\n"
        info_text += f"  Z: {true_land[2]:.1f}mm\n\n"
    
    if pred_info["triggered"] and pred_info["pred"] is not None:
        info_text += f"Predicted landing:\n"
        info_text += f"  X: {pred_info['pred'][0]:.1f}cm\n"
        info_text += f"  Y: {pred_info['pred'][1]:.1f}cm\n"
        info_text += f"  Z: {pred_info['pred'][2]:.1f}mm\n\n"
        
        info_text += f"Trigger frame: {pred_info['idx']}/{len(traj)}\n"
        info_text += f"Trigger time: {pred_info['idx']/FPS:.2f}s\n"
        info_text += f"Trigger point: ({pred_info.get('trigger_x', 0):.1f}, {pred_info.get('trigger_y', 0):.1f}, {pred_info.get('trigger_z', 0):.0f})\n"
        info_text += f"Predicted time remain: {pred_info['pred_remain']:.2f}s\n"
        
        if true_land is not None and pred_info["true_remain"] is not None:
            info_text += f"Actual time remain: {pred_info['true_remain']:.2f}s\n"
            info_text += f"Time error: {pred_info['dT']:.2f}s\n\n"
            
            info_text += f"XY error: {pred_info['errxy']:.2f}cm\n"
            info_text += f"Z error: {pred_info['errz']:.1f}mm"
    
    axes[1, 0].axis('off')
    axes[1, 0].text(0, 0.5, info_text, fontsize=10, verticalalignment='center', 
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[1, 1].axis('off')
    
    plt.suptitle(f"Landing Point Prediction - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Result saved to: {save_path}")

# ---------------- ROS2节点 ---------------- #
class LSTMProgressNode(Node):
    def __init__(self):
        super().__init__('lstm_progress_node')
        
        # 创建目标位置发布者（可选，向后兼容）
        self.target_pub = self.create_publisher(
            Pose2D,
            '/target_pos',
            10
        )
        
        # 创建触发点坐标发布者（可选，向后兼容）
        self.trigger_pub = self.create_publisher(
            Point,
            '/trigger_point',
            10
        )
        
        # 新增：创建Twist消息发布者（一次发送6个数据）
        self.twist_pub = self.create_publisher(
            Twist,
            '/gesture_twist_data',
            10
        )
        
        self.get_logger().info("✅ Publishers created:")
        self.get_logger().info("  - /target_pos (Pose2D): Predicted landing point (optional)")
        self.get_logger().info("  - /trigger_point (Point): Trigger point coordinates (optional)")
        self.get_logger().info("  - /gesture_twist_data (Twist): All data in Twist format (recommended)")
        self.get_logger().info("    linear.x: target_x, linear.y: target_y, linear.z: 0.0")
        self.get_logger().info("    angular.x: trigger_x, angular.y: trigger_y, angular.z: trigger_z")
        
        # 加载LSTM模型
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"Model file not found: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        try:
            self.model = LSTMProgressModel()
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            self.model.eval()
            self.get_logger().info(f"✅ LSTM model loaded from {MODEL_PATH}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise
        
        # 初始化RealSense
        self.get_logger().info("Initializing RealSense camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        try:
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.get_logger().info("✅ RealSense camera initialized successfully")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to start RealSense pipeline: {e}")
            raise
        
        # 初始化MediaPipe
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 状态变量
        self.traj = []           # 轨迹数据 [x, y, z]
        self.H = H               # 使用硬编码的单应性矩阵
        self.active = False      # 实验是否进行中
        self.frozen = False      # 是否已结束（显示结果）
        self.frozen_frame = None # 结束时的帧
        self.true_land = None    # 真实落点
        self.percent = 0.5
        self.extend = 1#0.5
        self.z_last=0
        self.verified=False
        
        # 预测信息
        self.pred_info = {
            "triggered": False,
            "pred": None,           # 预测落点 [x, y, z]
            "pred_remain": None,    # 预测剩余时间
            "win": None,            # 触发时的窗口数据
            "idx": None,            # 触发帧索引
            "true_remain": None,    # 实际剩余时间
            "dT": None,             # 时间误差
            "errxy": None,          # XY误差
            "errz": None,           # Z误差
            "trigger_x": None,      # 触发点x坐标
            "trigger_y": None,      # 触发点y坐标
            "trigger_z": None       # 触发点z坐标
        }
        
        # 运动检测相关
        self.started = False             # 是否开始运动
        self.moving_counter = 0          # 连续移动帧数
        
        # 发布标志
        self.prediction_published = False
        self.trigger_point_published = False
        self.twist_published = False
        
        # 创建定时器用于处理帧
        self.timer = self.create_timer(0.033, self.process_frame)  # ~30 FPS
        
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("LSTM Progress Trigger Prediction System")
        self.get_logger().info("="*60)
        self.get_logger().info("Controls:")
        self.get_logger().info("  's' - Start new experiment")
        self.get_logger().info("  'e' - End experiment and show results")
        self.get_logger().info("  SPACE - Quit program")
        self.get_logger().info("="*60)
        self.get_logger().info(f"Homography matrix loaded: OK")
        self.get_logger().info("="*60)
    
    def publish_target_position(self, x, y):
        """发布目标位置到/target_pos话题（可选）"""
        try:
            msg = Pose2D()
            msg.x = float(x)  # X坐标，单位：厘米
            msg.y = float(y)  # Y坐标，单位：厘米
            msg.theta = 0.0   # 第三个值随便取，这里取0.0
            
            self.target_pub.publish(msg)
            self.get_logger().info(f"📤 Published target position to /target_pos: ({msg.x:.1f}cm, {msg.y:.1f}cm)")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to publish target position: {e}")
            return False
    
    def publish_trigger_point(self, x, y, z):
        """发布触发点坐标到/trigger_point话题（可选）"""
        try:
            msg = Point()
            msg.x = float(x)  # X坐标，单位：厘米
            msg.y = float(y)  # Y坐标，单位：厘米
            msg.z = float(z)  # Z坐标，单位：毫米
            
            self.trigger_pub.publish(msg)
            self.get_logger().info(f"📤 Published trigger point to /trigger_point: ({msg.x:.1f}cm, {msg.y:.1f}cm, {msg.z:.1f}mm)")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to publish trigger point: {e}")
            return False
    
    def publish_twist_data(self, target_x, target_y, trigger_x, trigger_y, trigger_z):
        """
        发布组合数据到/gesture_twist_data话题
        使用Twist消息类型一次性发送5个数据：
        
        linear.x  = target_x     (预测点X坐标，单位：cm)
        linear.y  = target_y     (预测点Y坐标，单位：cm)
        linear.z  = 0.0          (保留字段，设为0.0)
        
        angular.x = trigger_x    (触发点X坐标，单位：cm)
        angular.y = trigger_y    (触发点Y坐标，单位：cm)
        angular.z = trigger_z    (触发点Z坐标，单位：mm)
        """
        try:
            msg = Twist()
            
            # linear部分：预测点坐标
            msg.linear.x = float(target_x)    # 预测点X坐标
            msg.linear.y = float(target_y)    # 预测点Y坐标
            msg.linear.z = 0.0                # 保留字段，设为0.0
            
            # angular部分：触发点坐标
            msg.angular.x = float(trigger_x)  # 触发点X坐标
            msg.angular.y = float(trigger_y)  # 触发点Y坐标
            msg.angular.z = float(trigger_z)  # 触发点Z坐标
            
            self.twist_pub.publish(msg)
            
            self.get_logger().info(f"🔄 Published Twist data to /gesture_twist_data:")
            self.get_logger().info(f"  Target point: ({target_x:.1f}cm, {target_y:.1f}cm)")
            self.get_logger().info(f"  Trigger point: ({trigger_x:.1f}cm, {trigger_y:.1f}cm, {trigger_z:.0f}mm)")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to publish Twist data: {e}")
            return False
    
    def process_frame(self):
        """处理每一帧"""
        try:
            # 获取帧
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            aligned = self.align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            
            if not color or not depth:
                return
        except Exception as e:
            self.get_logger().error(f"Frame acquisition error: {e}")
            return
        
        frame = np.asanyarray(color.get_data())
        show = frame.copy()
        
        # 手部检测
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        x = y = z = None
        cx = cy = None
        right_hand_found = False
        
        if results.multi_hand_landmarks and hasattr(results, 'multi_handedness'):
         for i, handedness in enumerate(results.multi_handedness):
           if handedness.classification[0].label == "Left": #mediapipe
            lm = results.multi_hand_landmarks[i].landmark[9]
            h, w = frame.shape[:2]
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))

            right_hand_found = True
            break
         if right_hand_found:
            # 获取深度
            try:
                z = depth.get_distance(cx, cy) * 1000.0
            except:
                z = None
            if z== 0:
                z=self.z_last
            else:
                self.z_last=z
            
            # 坐标转换
            if self.H is not None and z is not None:
                val = px2cm(self.H, cx, cy)
                if val is not None:
                    x, y = val
            
            # 在手上画点
            if cx is not None and cy is not None:
                cv2.circle(show, (cx, cy), 10, BLUE, -1)
                if x is not None and y is not None and z is not None:
                       cv2.putText(show, f"({x:.1f},{y:.1f},{z:.0f})", 
                                  (cx + 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, BLUE, 3)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):  # 开始新实验
            self.active = True
            self.frozen = False
            self.started = False
            self.moving_counter = 0
            self.traj.clear()
            self.frozen_frame = None
            self.true_land = None
            self.verified=False
            
            # 重置预测信息
            self.pred_info = {k: v for k, v in self.pred_info.items()}
            self.pred_info["triggered"] = False
            self.pred_info["pred"] = None
            self.pred_info["idx"] = None
            self.pred_info["trigger_x"] = None
            self.pred_info["trigger_y"] = None
            self.pred_info["trigger_z"] = None
            
            # 重置发布标志
            self.prediction_published = False
            self.trigger_point_published = False
            self.twist_published = False
            
            self.get_logger().info("\n" + "="*60)
            self.get_logger().info("New experiment started")
            self.get_logger().info("Make a pointing gesture towards the menu")
            self.get_logger().info("Press 'e' to end and see results")
            self.get_logger().info("="*60)
        
        elif key == ord('e') and self.active:  # 结束实验
            self.active = False
            self.frozen = True
            self.frozen_frame = show.copy()
            
            # 获取真实落点
            if len(self.traj) > 0:
                self.true_land = self.traj[-1]
                self.get_logger().info(f"\nTrue landing point: ({self.true_land[0]:.1f}, {self.true_land[1]:.1f}, {self.true_land[2]:.0f})")
            
            # 计算误差等信息
            if self.pred_info["triggered"] and self.true_land is not None and self.pred_info["pred"] is not None:
                N = len(self.traj)
                
                # 实际剩余时间
                if self.pred_info["idx"] is not None:
                    self.pred_info["true_remain"] = (N - self.pred_info["idx"]) / FPS
                    self.pred_info["dT"] = abs(self.pred_info["pred_remain"] - self.pred_info["true_remain"])
                
                # 空间误差
                self.pred_info["errxy"] = xy_err_cm(self.pred_info["pred"], self.true_land)
                self.pred_info["errz"] = z_err_mm(self.pred_info["pred"], self.true_land)
                
                # 显示结果
                self.get_logger().info("\n" + "="*60)
                self.get_logger().info("EXPERIMENT RESULTS")
                self.get_logger().info("="*60)
                self.get_logger().info(f"Total frames: {N}")
                self.get_logger().info(f"Total time: {N/FPS:.2f}s")
                self.get_logger().info(f"Trigger frame: {self.pred_info['idx']}/{N} ({self.pred_info['idx']/N*100:.1f}%)")
                self.get_logger().info(f"Trigger time: {self.pred_info['idx']/FPS:.2f}s")
                if self.pred_info["trigger_x"] is not None:
                    self.get_logger().info(f"Trigger point: ({self.pred_info['trigger_x']:.1f}, {self.pred_info['trigger_y']:.1f}, {self.pred_info['trigger_z']:.0f})")
                self.get_logger().info(f"Predicted time remain: {self.pred_info['pred_remain']:.2f}s")
                self.get_logger().info(f"Actual time remain: {self.pred_info['true_remain']:.2f}s")
                self.get_logger().info(f"Time error: {self.pred_info['dT']:.2f}s")
                self.get_logger().info(f"XY error: {self.pred_info['errxy']:.2f}cm")
                self.get_logger().info(f"Z error: {self.pred_info['errz']:.1f}mm")
                self.get_logger().info("="*60)
                
                # 保存结果图
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(SAVE_DIR, f"result_{timestamp}.png")
                save_result_plot(self.traj, self.true_land, self.pred_info, save_path)
            
            else:
                self.get_logger().info("\nExperiment ended but no prediction was triggered")
                self.get_logger().info("The model's progress never reached 0.5")
            
            # 在图像上绘制结果
            if self.frozen_frame is not None and self.H is not None:
                # 绘制真实落点
                if self.true_land is not None:
                    tp = cm2px(self.H, self.true_land[0], self.true_land[1])
                    if tp:
                        cv2.circle(self.frozen_frame, tp, 12, GREEN, -1)
                        cv2.putText(self.frozen_frame, "True", (tp[0] + 15, tp[1] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
                
                # 绘制预测落点
                if self.pred_info["pred"] is not None:
                    pp = cm2px(self.H, self.pred_info["pred"][0], self.pred_info["pred"][1])
                    if pp:
                        cv2.circle(self.frozen_frame, pp, 12, RED, -1)
                        cv2.putText(self.frozen_frame, "Pred", (pp[0] + 15, pp[1] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
        
        elif key == 32:  # 空格退出
            self.get_logger().info("\nExiting program...")
            self.cleanup()
            cv2.destroyAllWindows()
            rclpy.shutdown()
            return
        
        # --- 实验进行中 --- #
        if self.active and self.H is not None and x is not None and y is not None and z is not None and z > 0:
            # 记录轨迹
            self.traj.append([x, y, z])
            
            # 检测运动开始
            if len(self.traj) >= 2 and not self.started:
                if not is_static_xy(self.traj[-1], self.traj[-2]):
                    self.moving_counter += 1
                    if self.moving_counter >= MIN_MOVING_FRAMES and z < 650:
                        self.started = True
                        self.get_logger().info("Movement detected, starting prediction monitoring...")
                else:
                    self.moving_counter = 0
            
            # 预测触发
            if self.started and len(self.traj) >= WINDOW_LEN and not self.pred_info["triggered"]:
                # 准备输入数据
                window = np.array(self.traj[-WINDOW_LEN:])
                feat = compute_velocity_acceleration(window)
                feat = to_features(feat)
                x_in = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                
                # 模型预测
                with torch.no_grad():
                    pos, prog, t_rem = self.model(x_in)
                    current_progress = prog.item()
                    
                    # 检查是否触发
                    if current_progress >= self.percent:
                        self.pred_info["triggered"] = True
                        self.pred_info["pred"] = (pos.numpy()[0] * np.array([X_DEN, Y_DEN, Z_DEN])).tolist()
                        self.pred_info["pred_remain"] = float(t_rem.item())
                        self.pred_info["win"] = window.copy()
                        self.pred_info["idx"] = len(self.traj)
                        
                        # 保存触发点坐标
                        self.pred_info["trigger_x"] = x
                        self.pred_info["trigger_y"] = y
                        self.pred_info["trigger_z"] = z
                        
                        # 扩展预测点
                        self.pred_info["pred"][0] = x + self.extend * (self.pred_info["pred"][0] - x)
                        self.pred_info["pred"][1] = y + self.extend * (self.pred_info["pred"][1] - y)
                        # if (self.pred_info["pred"][1]>16 and  self.pred_info["pred"][1]<21 and self.pred_info["pred"][0]>35 and self.pred_info["pred"][0]<45):
                        #       self.verified=True
                        # if (self.pred_info["pred"][0]>35 and self.pred_info["pred"][0]<45):
                        #      self.verified=True
                        
                        self.get_logger().info(f"\n🚀 PREDICTION TRIGGERED!")
                        self.get_logger().info(f"  Frame: {len(self.traj)}")
                        self.get_logger().info(f"  Progress: {current_progress:.3f}")
                        self.get_logger().info(f"  Trigger point: ({x:.1f}, {y:.1f}, {z:.0f})")
                        self.get_logger().info(f"  Predicted landing: ({self.pred_info['pred'][0]:.1f}, {self.pred_info['pred'][1]:.1f}, {self.pred_info['pred'][2]:.0f})")
                        self.get_logger().info(f"  Predicted time remain: {self.pred_info['pred_remain']:.2f}s")
                        
                        # 发布Twist数据（一次发送所有数据）
                        if not self.twist_published :#and self.verified==True:
                            pred_x, pred_y, _ = self.pred_info["pred"]
                            if self.publish_twist_data(
                                pred_x, pred_y,  # 预测点坐标
                                x, y, z          # 触发点坐标
                            ):
                                self.twist_published = True
                        
                        # 可选：分别发布（向后兼容）
                        if not self.trigger_point_published:
                            if self.publish_trigger_point(x, y, z):
                                self.trigger_point_published = True
                        
                        if self.pred_info["pred"] is not None and not self.prediction_published:
                            pred_x, pred_y, _ = self.pred_info["pred"]
                            if self.publish_target_position(pred_x, pred_y):
                                self.prediction_published = True
            
            # 在实时画面中显示预测点
            if self.pred_info["triggered"] and self.pred_info["pred"] is not None and self.H is not None:
                pp = cm2px(self.H, self.pred_info["pred"][0], self.pred_info["pred"][1])
                if pp:
                    cv2.circle(show, pp, 12, RED, -1)
                    cv2.putText(show, "Predicted", (pp[0] + 15, pp[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
                    
        # 显示状态信息
        status_y = 30
        # if self.active:
        #     status_text = f"EXPERIMENT ACTIVE | Frames: {len(self.traj)}"
        #     if self.pred_info["triggered"]:
        #         status_text += " | DATA SENT"
        #         cv2.putText(show, status_text, (20, status_y), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            
        #     if self.started and not self.pred_info["triggered"]:
        #           cv2.putText(show, "Waiting for progress >= 0.5...", (20, status_y + 30), 
        #                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        #     if self.pred_info["triggered"]:
        #          cv2.putText(show, "DATA TRIGGERED & PUBLISHED", (20, status_y + 30), 
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
        # else:
        #     cv2.putText(show, "Press 's' to start experiment", (20, status_y), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        
        # 显示H矩阵状态
        # if self.H is not None:
        #     cv2.putText(show, "H: LOADED", (show.shape[1] - 120, 30), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
        
        # 显示ROS2发布状态
        ros_status_x = show.shape[1] - 300
        ros_status_y = 60
        
        # 检查订阅者数量
        target_subs = self.target_pub.get_subscription_count()
        trigger_subs = self.trigger_pub.get_subscription_count()
        twist_subs = self.twist_pub.get_subscription_count()
        
        if target_subs > 0 or trigger_subs > 0 or twist_subs > 0:
            status_parts = []
            if target_subs > 0:
                status_parts.append(f"Target: {target_subs}")
            if trigger_subs > 0:
                status_parts.append(f"Trigger: {trigger_subs}")
            # if twist_subs > 0:
            #     status_parts.append(f"Subscriber:1")
            if twist_subs > 0:
                status_parts.append(f"Twist: {twist_subs}")
    
            # status_line = f"ROS2: " + " | ".join(status_parts)
            # cv2.putText(show, status_line, (ros_status_x, ros_status_y), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
        else:
            cv2.putText(show, "ROS2: No subscribers", (ros_status_x, ros_status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        
        # 显示当前帧
        display_frame = self.frozen_frame if self.frozen and self.frozen_frame is not None else show
        #cv2.imshow("LSTM Progress Trigger Prediction", display_frame)
        cv2.imshow("Landing Point Prediction", display_frame)
        # 检查窗口是否被关闭
        if cv2.getWindowProperty("Landing Point Prediction", cv2.WND_PROP_VISIBLE) < 1:
            self.get_logger().info("\nWindow closed, shutting down...")
            self.cleanup()
            cv2.destroyAllWindows()
            rclpy.shutdown()
            return
    
    def cleanup(self):
        """清理资源"""
        self.get_logger().info("Cleaning up resources...")
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        if hasattr(self, 'hands'):
            self.hands.close()
        self.get_logger().info("Resources cleaned up")
    
    def destroy_node(self):
        """销毁节点"""
        self.cleanup()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    # 创建OpenCV窗口
    cv2.namedWindow("Landing Point Prediction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Landing Point Prediction", 1280, 720)
    
    node = LSTMProgressNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()