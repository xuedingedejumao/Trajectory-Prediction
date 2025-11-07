"""
无迹卡尔曼滤波器（UKF）实现
用于非线性系统的状态估计，比KF更适合处理机动目标
"""

import numpy as np


class UnscentedKalmanFilter:
    """
    无迹卡尔曼滤波器

    状态向量: [x, y, z, vx, vy, vz]
    观测向量: [x, y, z, vx, vy, vz]
    """

    def __init__(self, dt=1.0, alpha=0.001, beta=2, kappa=0):
        """
        初始化UKF

        参数:
            dt: 时间步长
            alpha: sigma点分布参数 (通常 0.001 <= alpha <= 1)
            beta: 先验知识参数 (高斯分布beta=2最优)
            kappa: 次要缩放参数
        """
        self.dt = dt
        self.dim_x = 6  # 状态维度
        self.dim_z = 6  # 观测维度

        # UKF参数
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # 计算lambda
        self.lambda_ = alpha ** 2 * (self.dim_x + kappa) - self.dim_x

        # 权重计算
        self.Wm, self.Wc = self._calculate_weights()

        # 过程噪声协方差矩阵 Q
        q = 0.5  # UKF可以用更大的过程噪声来处理非线性
        self.Q = np.array([
            [dt ** 4 / 4, 0, 0, dt ** 3 / 2, 0, 0],
            [0, dt ** 4 / 4, 0, 0, dt ** 3 / 2, 0],
            [0, 0, dt ** 4 / 4, 0, 0, dt ** 3 / 2],
            [dt ** 3 / 2, 0, 0, dt ** 2, 0, 0],
            [0, dt ** 3 / 2, 0, 0, dt ** 2, 0],
            [0, 0, dt ** 3 / 2, 0, 0, dt ** 2]
        ]) * q

        # 观测噪声协方差矩阵 R
        self.R = np.diag([4.0, 4.0, 4.0, 1.0, 1.0, 1.0])

        # 状态估计
        self.x = np.zeros(6)

        # 估计协方差矩阵
        self.P = np.eye(6) * 100

        # 是否已初始化
        self.initialized = False

    def _calculate_weights(self):
        """
        计算sigma点的权重

        返回:
            Wm: 均值权重
            Wc: 协方差权重
        """
        n = self.dim_x
        lambda_ = self.lambda_

        # 2n+1个sigma点
        num_sigmas = 2 * n + 1

        # 均值权重
        Wm = np.zeros(num_sigmas)
        Wm[0] = lambda_ / (n + lambda_)
        Wm[1:] = 0.5 / (n + lambda_)

        # 协方差权重
        Wc = Wm.copy()
        Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha ** 2 + self.beta)

        return Wm, Wc

    def _generate_sigma_points(self, x, P):
        """
        生成sigma点

        参数:
            x: 状态均值
            P: 状态协方差

        返回:
            sigma点矩阵 (2n+1, n)
        """
        n = self.dim_x
        lambda_ = self.lambda_

        # 计算矩阵平方根 (Cholesky分解)
        try:
            A = np.linalg.cholesky((n + lambda_) * P)
        except np.linalg.LinAlgError:
            # 如果P不是正定的，添加小的正则化项
            P_reg = P + np.eye(n) * 1e-6
            A = np.linalg.cholesky((n + lambda_) * P_reg)

        # 生成sigma点
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x

        for i in range(n):
            sigmas[i + 1] = x + A[i]
            sigmas[n + i + 1] = x - A[i]

        return sigmas

    def _state_transition(self, x):
        """
        状态转移函数 (可以是非线性的)
        这里使用恒定加速度模型

        参数:
            x: 状态向量 [x, y, z, vx, vy, vz]

        返回:
            预测的状态向量
        """
        dt = self.dt
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        return F @ x

    def _observation_function(self, x):
        """
        观测函数 (可以是非线性的)
        这里假设直接观测所有状态

        参数:
            x: 状态向量

        返回:
            观测向量
        """
        return x.copy()

    def initialize(self, z):
        """
        使用首次观测初始化滤波器

        参数:
            z: 观测向量
        """
        self.x = z.copy()
        self.initialized = True

    def predict(self):
        """
        UKF预测步骤

        返回:
            预测的状态向量
        """
        # 1. 生成sigma点
        sigmas = self._generate_sigma_points(self.x, self.P)

        # 2. 通过状态转移函数传播sigma点
        sigmas_f = np.zeros_like(sigmas)
        for i in range(sigmas.shape[0]):
            sigmas_f[i] = self._state_transition(sigmas[i])

        # 3. 计算预测均值
        self.x = np.sum(self.Wm[:, np.newaxis] * sigmas_f, axis=0)

        # 4. 计算预测协方差
        self.P = self.Q.copy()
        for i in range(sigmas_f.shape[0]):
            y = sigmas_f[i] - self.x
            self.P += self.Wc[i] * np.outer(y, y)

        return self.x.copy()

    def update(self, z):
        """
        UKF更新步骤

        参数:
            z: 观测向量

        返回:
            更新后的状态向量
        """
        # 1. 生成sigma点
        sigmas = self._generate_sigma_points(self.x, self.P)

        # 2. 通过观测函数传播sigma点
        sigmas_h = np.zeros((sigmas.shape[0], self.dim_z))
        for i in range(sigmas.shape[0]):
            sigmas_h[i] = self._observation_function(sigmas[i])

        # 3. 计算观测预测均值
        z_pred = np.sum(self.Wm[:, np.newaxis] * sigmas_h, axis=0)

        # 4. 计算观测协方差和互协方差
        P_zz = self.R.copy()  # 观测协方差
        P_xz = np.zeros((self.dim_x, self.dim_z))  # 互协方差

        for i in range(sigmas.shape[0]):
            z_diff = sigmas_h[i] - z_pred
            x_diff = sigmas[i] - self.x

            P_zz += self.Wc[i] * np.outer(z_diff, z_diff)
            P_xz += self.Wc[i] * np.outer(x_diff, z_diff)

        # 5. 计算卡尔曼增益
        K = P_xz @ np.linalg.inv(P_zz)

        # 6. 更新状态和协方差
        y = z - z_pred  # 新息
        self.x = self.x + K @ y
        self.P = self.P - K @ P_zz @ K.T

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