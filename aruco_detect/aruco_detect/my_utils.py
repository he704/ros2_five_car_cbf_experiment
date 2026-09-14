#!/usr/bin/env python3
"""
utils.py - 迁移到现代transformations替代方案
替换旧的tf.transformations依赖
"""

import math
import numpy as np
import time      
import cv2 

PI = 3.1415926

# 尝试导入现代替代方案
try:
    # 方案1: 使用tf_transformations (ROS2推荐)
    from tf_transformations import (
        euler_matrix, quaternion_from_matrix, quaternion_from_euler,
        euler_from_quaternion, quaternion_matrix, inverse_matrix,
        quaternion_from_euler, euler_from_quaternion
    )
    TRANSFORM_METHOD = "tf_transformations"
except ImportError:
    try:
        # 方案2: 使用scipy
        from scipy.spatial.transform import Rotation as R
        TRANSFORM_METHOD = "scipy"
        print("使用scipy进行旋转转换")
    except ImportError:
        # 方案3: 手动实现所有功能
        TRANSFORM_METHOD = "manual"
        print("使用手动实现的旋转转换")

def euler_to_matrix_rad(x, y, z, axes="sxyz"):
    """将欧拉角转换为4x4齐次变换矩阵"""
    if TRANSFORM_METHOD == "tf_transformations":
        return euler_matrix(x, y, z, axes)
    elif TRANSFORM_METHOD == "scipy":
        rot = R.from_euler(axes[:3], [x, y, z])
        matrix = np.eye(4)
        matrix[:3, :3] = rot.as_matrix()
        return matrix
    else:  # manual
        # 手动实现欧拉角到矩阵的转换
        # 简化版本，仅支持'sxyz'顺序
        cx, sx = math.cos(x), math.sin(x)
        cy, sy = math.cos(y), math.sin(y)  
        cz, sz = math.cos(z), math.sin(z)
        
        matrix = np.eye(4)
        matrix[0, 0] = cy * cz
        matrix[0, 1] = -cy * sz
        matrix[0, 2] = sy
        matrix[1, 0] = sx * sy * cz + cx * sz
        matrix[1, 1] = -sx * sy * sz + cx * cz
        matrix[1, 2] = -sx * cy
        matrix[2, 0] = -cx * sy * cz + sx * sz
        matrix[2, 1] = cx * sy * sz + sx * cz
        matrix[2, 2] = cx * cy
        return matrix

def matrix_to_euler_rad(matrix):
    """将4x4齐次变换矩阵转换为欧拉角"""
    if TRANSFORM_METHOD == "tf_transformations":
        q = quaternion_from_matrix(matrix)
        return euler_from_quaternion(q, axes='sxyz')
    elif TRANSFORM_METHOD == "scipy":
        rot = R.from_matrix(matrix[:3, :3])
        return rot.as_euler('xyz')
    else:  # manual
        # 手动实现矩阵到欧拉角的转换
        sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(matrix[2, 1], matrix[2, 2])
            y = math.atan2(-matrix[2, 0], sy)
            z = math.atan2(matrix[1, 0], matrix[0, 0])
        else:
            x = math.atan2(-matrix[1, 2], matrix[1, 1])
            y = math.atan2(-matrix[2, 0], sy)
            z = 0
            
        return np.array([x, y, z])

def matrix_to_quaternion(matrix):
    """将4x4齐次变换矩阵转换为四元数"""
    if TRANSFORM_METHOD == "tf_transformations":
        return quaternion_from_matrix(matrix)
    elif TRANSFORM_METHOD == "scipy":
        rot = R.from_matrix(matrix[:3, :3])
        return rot.as_quat()
    else:  # manual
        # 手动实现矩阵到四元数的转换
        tr = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (matrix[2, 1] - matrix[1, 2]) / S
            qy = (matrix[0, 2] - matrix[2, 0]) / S 
            qz = (matrix[1, 0] - matrix[0, 1]) / S
        elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
            S = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            qw = (matrix[2, 1] - matrix[1, 2]) / S
            qx = 0.25 * S
            qy = (matrix[0, 1] + matrix[1, 0]) / S
            qz = (matrix[0, 2] + matrix[2, 0]) / S
        elif matrix[1, 1] > matrix[2, 2]:
            S = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            qw = (matrix[0, 2] - matrix[2, 0]) / S
            qx = (matrix[0, 1] + matrix[1, 0]) / S
            qy = 0.25 * S
            qz = (matrix[1, 2] + matrix[2, 1]) / S
        else:
            S = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            qw = (matrix[1, 0] - matrix[0, 1]) / S
            qx = (matrix[0, 2] + matrix[2, 0]) / S
            qy = (matrix[1, 2] + matrix[2, 1]) / S
            qz = 0.25 * S
            
        return np.array([qx, qy, qz, qw])

def axis_to_quaternion(axis_angle):
    """
    将轴角表示的旋转转换为四元数表示
    """
    if len(axis_angle) == 3:
        theta = np.linalg.norm(axis_angle)
        axis = axis_angle / theta
        qw = np.cos(theta / 2)
        qx = axis[0] * np.sin(theta / 2)
        qy = axis[1] * np.sin(theta / 2)
        qz = axis[2] * np.sin(theta / 2)
        return np.array([qx, qy, qz, qw])

    elif len(axis_angle) == 4:
        q = [
            np.sin(axis_angle[3]/2) * axis_angle[0],
            np.sin(axis_angle[3]/2) * axis_angle[1],
            np.sin(axis_angle[3]/2) * axis_angle[2],
            np.cos(axis_angle[3]/2)
        ]
        return np.array(q)
    else:
        print("Wrong Axis Angle!!")
        return None

def quaternion_to_matrix(quat):
    """将四元数转换为4x4齐次变换矩阵"""
    if TRANSFORM_METHOD == "tf_transformations":
        return quaternion_matrix(quat)
    elif TRANSFORM_METHOD == "scipy":
        rot = R.from_quat(quat)
        matrix = np.eye(4)
        matrix[:3, :3] = rot.as_matrix()
        return matrix
    else:  # manual
        # 手动实现四元数到矩阵的转换
        qx, qy, qz, qw = quat
        
        matrix = np.eye(4)
        matrix[0, 0] = 1 - 2*qy*qy - 2*qz*qz
        matrix[0, 1] = 2*qx*qy - 2*qz*qw
        matrix[0, 2] = 2*qx*qz + 2*qy*qw
        
        matrix[1, 0] = 2*qx*qy + 2*qz*qw
        matrix[1, 1] = 1 - 2*qx*qx - 2*qz*qz
        matrix[1, 2] = 2*qy*qz - 2*qx*qw
        
        matrix[2, 0] = 2*qx*qz - 2*qy*qw
        matrix[2, 1] = 2*qy*qz + 2*qx*qw  
        matrix[2, 2] = 1 - 2*qx*qx - 2*qy*qy
        
        return matrix

def quaternion_to_euler_rad(quat, axes='sxyz'):
    """将四元数转换为欧拉角"""
    if TRANSFORM_METHOD == "tf_transformations":
        return euler_from_quaternion(quat, axes=axes)
    elif TRANSFORM_METHOD == "scipy":
        rot = R.from_quat(quat)
        return rot.as_euler(axes[:3])
    else:  # manual
        # 通过矩阵转换
        matrix = quaternion_to_matrix(quat)
        return matrix_to_euler_rad(matrix)

def axis_to_euler_rad(axis_angle, axes='sxyz'):
    """将轴角转换为欧拉角"""
    q = axis_to_quaternion(axis_angle)
    return quaternion_to_euler_rad(q, axes=axes)

def axis_to_matrix(axis_angle):
    """将轴角转换为4x4齐次变换矩阵"""
    q = axis_to_quaternion(axis_angle)
    return quaternion_to_matrix(q)

def euler_to_quaternion_rad(x, y, z):
    """将欧拉角转换为四元数"""
    if TRANSFORM_METHOD == "tf_transformations":
        return quaternion_from_euler(x, y, z, axes='sxyz')
    elif TRANSFORM_METHOD == "scipy":
        rot = R.from_euler('xyz', [x, y, z])
        return rot.as_quat()
    else:  # manual
        # 手动实现欧拉角到四元数的转换
        cy = math.cos(y * 0.5)
        sy = math.sin(y * 0.5)
        cp = math.cos(x * 0.5)
        sp = math.sin(x * 0.5)
        cr = math.cos(z * 0.5)
        sr = math.sin(z * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw])

def rad_to_degree(rad):
    """弧度转角度"""
    return rad / math.pi * 180

def degree_to_euler(degree):
    """角度转弧度"""
    return degree / 180 * math.pi

def inverse_matrix(matrix):
    """求矩阵的逆"""
    if TRANSFORM_METHOD == "tf_transformations":
        return inverse_matrix(matrix)
    else:
        return np.linalg.inv(matrix)

def dot_matrix(a, b):
    """矩阵乘法"""
    return np.dot(a, b)

def PoseToRTmatrix(euler, T, axes="sxyz"):
    """将位姿(欧拉角+平移)转换为4x4齐次变换矩阵"""
    R_matrix = euler_to_matrix_rad(euler[0], euler[1], euler[2], axes=axes)
    R_matrix[0][3] = T[0]
    R_matrix[1][3] = T[1]
    R_matrix[2][3] = T[2]
    return R_matrix

def RTmatrixToPose(matrix):
    """将4x4齐次变换矩阵转换为位姿"""
    x, y, z = matrix_to_euler_rad(matrix)
    Tx = matrix[0][3] 
    Ty = matrix[1][3]
    Tz = matrix[2][3]
    return x, y, z, Tx, Ty, Tz

def Q_PoseToRTmatrix(T, quat):
    """将位姿(四元数+平移)转换为4x4齐次变换矩阵"""
    matrix = quaternion_to_matrix(quat)
    matrix[0][3] = T[0]
    matrix[1][3] = T[1]
    matrix[2][3] = T[2]
    return matrix

def Q_RTmatrixToPose(matrix):
    """将4x4齐次变换矩阵转换为位姿(四元数+平移)"""
    quat = matrix_to_quaternion(matrix)
    T = [0, 0, 0]
    T[0] = matrix[0][3] 
    T[1] = matrix[1][3]
    T[2] = matrix[2][3]
    return T, quat

# 测试代码
if __name__ == "__main__":
    # 测试基本功能
    print(f"使用的转换方法: {TRANSFORM_METHOD}")
    
    # 测试欧拉角到矩阵的转换
    euler = [0.1, 0.2, 0.3]
    matrix = euler_to_matrix_rad(*euler)
    print("欧拉角到矩阵测试通过")
    
    # 测试矩阵到欧拉角的转换
    euler_back = matrix_to_euler_rad(matrix)
    print("矩阵到欧拉角测试通过")
    
    # 测试四元数转换
    quat = euler_to_quaternion_rad(*euler)
    print("四元数转换测试通过")
    
    print("所有功能测试完成!")