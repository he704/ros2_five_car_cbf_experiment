import cv2
import numpy as np

rvec_obj =np.array([0.34 , -3.13 , 0.083])
rvec_obj = rvec_obj.squeeze()
print(rvec_obj)
rotation_matrix, _ = cv2.Rodrigues(rvec_obj)
# 提取绕Z轴的角度（偏航角yaw）
theta_z_test = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
theta_z_test=np.degrees(theta_z_test)
print(rotation_matrix)
print(theta_z_test)
