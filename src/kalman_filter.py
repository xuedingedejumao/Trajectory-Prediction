"""
卡尔曼滤波器（KF）实现
用于线性系统的状态估计
"""

import numpy as np


class KalmanFilter:
    """
    标准卡尔曼滤波器

    状态向量: [x, y, z, vx, vy, vz]
    观测向量: [x, y, z, vx, vy, vz]
    """

    def __init__(self, dt=1.0):
        """
        初始化卡尔曼滤波器

        参数:
            dt: 时间步长
        """
        self.dt = dt
        self.dim_x = 6  # 状态维度 (x, y, z, vx, vy, vz)
        self.dim_z = 6  # 观测维度

        # 状态转移矩阵 F (匀速运动模型)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # 观测矩阵 H (直接观测所有状态)
        self.H = np.eye(6)

        # 过程噪声协方差矩阵 Q
        q = 0.1  # 过程噪声强度
        self.Q = np.array([
            [dt ** 4 / 4, 0, 0, dt ** 3 / 2, 0, 0],
            [0, dt ** 4 / 4, 0, 0, dt ** 3 / 2, 0],
            [0, 0, dt ** 4 / 4, 0, 0, dt ** 3 / 2],
            [dt ** 3 / 2, 0, 0, dt ** 2, 0, 0],
            [0, dt ** 3 / 2, 0, 0, dt ** 2, 0],
            [0, 0, dt ** 3 / 2, 0, 0, dt ** 2]
        ]) * q

        # 观测噪声协方差矩阵 R
        self.R = np.diag([4.0, 4.0, 4.0, 1.0, 1.0, 1.0])  # 位置和速度的测量噪声

        # 状态估计
        self.x = np.zeros(6)

        # 估计协方差矩阵
        self.P = np.eye(6) * 100  # 初始不确定性较大

        # 是否已初始化
        self.initialized = False

    def initialize(self, z):
        """
        使用首次观测初始化滤波器

        参数:
            z: 观测向量 [x, y, z, vx, vy, vz]
        """
        self.x = z.copy()
        self.initialized = True

    def predict(self):
        """
        预测步骤：根据运动模型预测下一时刻的状态

        返回:
            预测的状态向量
        """
        # 状态预测: x_k|k-1 = F * x_k-1|k-1
        self.x = self.F @ self.x

        # 协方差预测: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(self, z):
        """
        更新步骤：使用观测值修正预测

        参数:
            z: 观测向量 [x, y, z, vx, vy, vz]

        返回:
            更新后的状态向量
        """
        # 计算新息（观测残差）: y = z - H * x_k|k-1
        y = z - self.H @ self.x

        # 新息协方差: S = H * P_k|k-1 * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # 卡尔曼增益: K = P_k|k-1 * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 状态更新: x_k|k = x_k|k-1 + K * y
        self.x = self.x + K @ y

        # 协方差更新: P_k|k = (I - K * H) * P_k|k-1
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.P = I_KH @ self.P

        return self.x.copy()

    def process(self, z):
        """
        处理一次观测（预测+更新）

        参数:
            z: 观测向量，如果为None则只进行预测

        返回:
            估计的状态向量
        """
        if not self.initialized:
            if z is not None:
                self.initialize(z)
            return self.x.copy()

        # 预测
        self.predict()

        # 如果有观测，则更新
        if z is not None:
            self.update(z)

        return self.x.copy()

    def get_state(self):
        """获取当前状态估计"""
        return self.x.copy()

    def get_covariance(self):
        """获取当前协方差矩阵"""
        return self.P.copy()