"""
数据生成模块
生成舰船和拦截弹的运动轨迹数据，并保存为CSV文件
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_ship_trajectory(num_frames=100, dt=1.0):
    """
    生成舰船运动轨迹

    参数:
        num_frames: 生成的帧数
        dt: 时间步长(秒)

    返回:
        DataFrame: 包含舰船轨迹数据
    """
    # 初始状态
    x0, y0, z0 = 0.0, 0.0, 0.0  # 初始位置 (米)
    vx0, vy0, vz0 = 50.0, 30.0, 0.0  # 初始速度 (米/秒)

    # 存储数据
    data = []

    # 当前状态
    x, y, z = x0, y0, z0
    vx, vy, vz = vx0, vy0, vz0

    # 加速度（模拟转向机动）
    ax = 0.5  # x方向加速度
    ay = -0.3  # y方向加速度
    az = 0.0  # z方向保持水平

    for frame in range(num_frames):
        # 更新速度（考虑加速度）
        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        # 更新位置
        x += vx * dt + 0.5 * ax * dt ** 2
        y += vy * dt + 0.5 * ay * dt ** 2
        z += vz * dt + 0.5 * az * dt ** 2

        # 添加随机噪声（模拟测量误差）
        noise_pos = np.random.normal(0, 2, 3)  # 位置噪声
        noise_vel = np.random.normal(0, 0.5, 3)  # 速度噪声

        # 模拟探测概率（90%的探测率）
        detected = 1 if np.random.rand() > 0.1 else -1

        # 记录数据
        data.append({
            'frameID': frame,
            'target_x': x + noise_pos[0],
            'target_y': y + noise_pos[1],
            'target_z': z + noise_pos[2],
            'target_Vx': vx + noise_vel[0],
            'target_Vy': vy + noise_vel[1],
            'target_Vz': vz + noise_vel[2],
            'detected': detected
        })

        # 周期性改变加速度（模拟机动）
        if frame % 30 == 0 and frame > 0:
            ax = np.random.uniform(-1, 1)
            ay = np.random.uniform(-1, 1)

    return pd.DataFrame(data)


def generate_interceptor_trajectory(num_frames=100, dt=1.0):
    """
    生成拦截弹运动轨迹

    参数:
        num_frames: 生成的帧数
        dt: 时间步长(秒)

    返回:
        DataFrame: 包含拦截弹轨迹数据
    """
    # 初始状态（从不同位置发射）
    x0, y0, z0 = -1000.0, -500.0, 100.0  # 初始位置 (米)
    vx0, vy0, vz0 = 150.0, 80.0, -2.0  # 初始速度 (米/秒)

    # 存储数据
    data = []

    # 当前状态
    x, y, z = x0, y0, z0
    vx, vy, vz = vx0, vy0, vz0

    # 拦截弹的加速度（模拟制导）
    ax = 2.0
    ay = 1.5
    az = -0.5

    for frame in range(num_frames):
        # 更新速度
        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        # 更新位置
        x += vx * dt + 0.5 * ax * dt ** 2
        y += vy * dt + 0.5 * ay * dt ** 2
        z += vz * dt + 0.5 * az * dt ** 2

        # 添加随机噪声
        noise_pos = np.random.normal(0, 3, 3)  # 拦截弹噪声稍大
        noise_vel = np.random.normal(0, 1, 3)

        # 模拟探测概率（85%的探测率）
        detected = 1 if np.random.rand() > 0.15 else -1

        # 记录数据
        data.append({
            'frameID': frame,
            'target_x': x + noise_pos[0],
            'target_y': y + noise_pos[1],
            'target_z': z + noise_pos[2],
            'target_Vx': vx + noise_vel[0],
            'target_Vy': vy + noise_vel[1],
            'target_Vz': vz + noise_vel[2],
            'detected': detected
        })

        # 周期性调整制导
        if frame % 20 == 0 and frame > 0:
            ax = np.random.uniform(-2, 3)
            ay = np.random.uniform(-2, 3)
            az = np.random.uniform(-1, 0.5)

    return pd.DataFrame(data)


def main():
    """主函数：生成数据并保存为CSV"""
    # 创建data目录
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    # 生成舰船轨迹
    print("生成舰船轨迹数据...")
    ship_data = generate_ship_trajectory(num_frames=100, dt=1.0)
    ship_file = data_dir / 'ship_trajectory.csv'
    ship_data.to_csv(ship_file, index=False)
    print(f"舰船轨迹已保存到: {ship_file}")
    print(f"探测帧数: {(ship_data['detected'] == 1).sum()}/{len(ship_data)}")

    # 生成拦截弹轨迹
    print("\n生成拦截弹轨迹数据...")
    interceptor_data = generate_interceptor_trajectory(num_frames=100, dt=1.0)
    interceptor_file = data_dir / 'interceptor_trajectory.csv'
    interceptor_data.to_csv(interceptor_file, index=False)
    print(f"拦截弹轨迹已保存到: {interceptor_file}")
    print(f"探测帧数: {(interceptor_data['detected'] == 1).sum()}/{len(interceptor_data)}")

    print("\n数据生成完成！")


if __name__ == '__main__':
    main()