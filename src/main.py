"""
主程序
运行完整的目标跟踪演示流程
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.kalman_filter import KalmanFilter
from src.unscented_kalman_filter import UnscentedKalmanFilter
from src.visualization import plot_comparison
from src.generate_data import main as generate_data


def load_data(filepath):
    """
    加载CSV数据

    参数:
        filepath: CSV文件路径

    返回:
        observations: 观测数据数组
        detections: 探测标志数组
    """
    df = pd.read_csv(filepath)

    # 提取观测数据 [x, y, z, vx, vy, vz]
    observations = df[['target_x', 'target_y', 'target_z',
                       'target_Vx', 'target_Vy', 'target_Vz']].values

    # 提取探测标志
    detections = df['detected'].values

    return observations, detections


def run_kf_tracking(observations, detections, dt=1.0):
    """
    运行卡尔曼滤波跟踪

    参数:
        observations: 观测数据
        detections: 探测标志
        dt: 时间步长

    返回:
        estimates: 估计结果数组
    """
    kf = KalmanFilter(dt=dt)
    estimates = []

    print("\n运行卡尔曼滤波器...")
    for i, (obs, det) in enumerate(zip(observations, detections)):
        # 如果目标被探测到，使用观测值；否则只进行预测
        if det == 1:
            state = kf.process(obs)
        else:
            state = kf.process(None)  # 未探测到时只预测

        estimates.append(state)

        if (i + 1) % 20 == 0:
            print(f"  处理帧 {i + 1}/{len(observations)}")

    return np.array(estimates)


def run_ukf_tracking(observations, detections, dt=1.0,
                     alpha=0.001, beta=2, kappa=0):
    """
    运行无迹卡尔曼滤波跟踪

    参数:
        observations: 观测数据
        detections: 探测标志
        dt: 时间步长
        alpha, beta, kappa: UKF参数

    返回:
        estimates: 估计结果数组
    """
    ukf = UnscentedKalmanFilter(dt=dt, alpha=alpha, beta=beta, kappa=kappa)
    estimates = []

    print(f"\n运行无迹卡尔曼滤波器 (alpha={alpha}, beta={beta})...")
    for i, (obs, det) in enumerate(zip(observations, detections)):
        # 如果目标被探测到，使用观测值；否则只进行预测
        if det == 1:
            state = ukf.process(obs)
        else:
            state = ukf.process(None)

        estimates.append(state)

        if (i + 1) % 20 == 0:
            print(f"  处理帧 {i + 1}/{len(observations)}")

    return np.array(estimates)


def main():
    """主函数"""
    print("=" * 60)
    print("目标跟踪系统 - KF/UKF演示")
    print("=" * 60)

    # 创建必要的目录
    Path('data').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    # 步骤1: 生成数据（如果不存在）
    ship_file = Path('data/ship_trajectory.csv')
    inter_file = Path('data/interceptor_trajectory.csv')

    if not ship_file.exists() or not inter_file.exists():
        print("\n步骤1: 生成轨迹数据...")
        generate_data()
    else:
        print("\n步骤1: 使用现有轨迹数据")

    # 步骤2: 加载数据
    print("\n步骤2: 加载数据...")
    ship_obs, ship_det = load_data(ship_file)
    inter_obs, inter_det = load_data(inter_file)
    print(f"  舰船数据: {len(ship_obs)} 帧, 探测率: {(ship_det == 1).sum() / len(ship_det) * 100:.1f}%")
    print(f"  拦截弹数据: {len(inter_obs)} 帧, 探测率: {(inter_det == 1).sum() / len(inter_det) * 100:.1f}%")

    # 步骤3: 舰船跟踪 - KF
    print("\n步骤3: 舰船跟踪 - 卡尔曼滤波器")
    ship_kf_est = run_kf_tracking(ship_obs, ship_det, dt=1.0)

    # 步骤4: 舰船跟踪 - UKF
    print("\n步骤4: 舰船跟踪 - 无迹卡尔曼滤波器")
    ship_ukf_est = run_ukf_tracking(ship_obs, ship_det, dt=1.0, alpha=0.001, beta=2)

    # 步骤5: 拦截弹跟踪 - UKF（两种参数配置）
    print("\n步骤5: 拦截弹跟踪 - 无迹卡尔曼滤波器")
    print("  配置1: 标准参数")
    inter_ukf_est1 = run_ukf_tracking(inter_obs, inter_det, dt=1.0,
                                      alpha=0.001, beta=2, kappa=0)

    print("  配置2: 调整参数以处理更高的非线性")
    inter_ukf_est2 = run_ukf_tracking(inter_obs, inter_det, dt=1.0,
                                      alpha=0.01, beta=2, kappa=1)

    # 步骤6: 可视化结果
    print("\n步骤6: 生成可视化结果...")
    plot_comparison(
        ship_obs, ship_kf_est, ship_ukf_est,
        inter_obs, inter_ukf_est1, inter_ukf_est2
    )

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"结果保存在 'results/' 目录")
    print(f"- ship_tracking.png: 舰船跟踪结果")
    print(f"- interceptor_tracking.png: 拦截弹跟踪结果")
    print("\n提示:")
    print("- KF适用于线性运动模型")
    print("- UKF能够更好地处理非线性和机动目标")
    print("- 调整UKF的alpha参数可以改变sigma点的分布")
    print("- 当目标未被探测到时，滤波器会使用运动模型进行预测")


if __name__ == '__main__':
    main()