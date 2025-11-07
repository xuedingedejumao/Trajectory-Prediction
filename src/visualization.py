"""
可视化模块
用于绘制轨迹对比和滤波效果
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_tracking_results(measurements, kf_estimates, ukf_estimates,
                          title="目标跟踪结果", save_path=None):
    """
    绘制跟踪结果对比图

    参数:
        measurements: 观测数据 (n, 6)
        kf_estimates: KF估计结果 (n, 6)
        ukf_estimates: UKF估计结果 (n, 6)
        title: 图表标题
        save_path: 保存路径（可选）
    """
    fig = plt.figure(figsize=(18, 12))

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 3D轨迹图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(measurements[:, 0], measurements[:, 1], measurements[:, 2],
             'b.', alpha=0.5, label='观测值', markersize=3)
    ax1.plot(kf_estimates[:, 0], kf_estimates[:, 1], kf_estimates[:, 2],
             'r-', linewidth=2, label='KF估计')
    ax1.plot(ukf_estimates[:, 0], ukf_estimates[:, 1], ukf_estimates[:, 2],
             'g-', linewidth=2, label='UKF估计')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D轨迹对比')
    ax1.legend()
    ax1.grid(True)

    # 2. X-Y平面投影
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(measurements[:, 0], measurements[:, 1],
             'b.', alpha=0.5, label='观测值', markersize=3)
    ax2.plot(kf_estimates[:, 0], kf_estimates[:, 1],
             'r-', linewidth=2, label='KF估计')
    ax2.plot(ukf_estimates[:, 0], ukf_estimates[:, 1],
             'g-', linewidth=2, label='UKF估计')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('X-Y平面投影')
    ax2.legend()
    ax2.grid(True)

    # 3. 位置误差对比 (相对于观测值)
    ax3 = fig.add_subplot(2, 3, 3)
    frames = np.arange(len(measurements))
    kf_pos_error = np.sqrt(np.sum((kf_estimates[:, :3] - measurements[:, :3])**2, axis=1))
    ukf_pos_error = np.sqrt(np.sum((ukf_estimates[:, :3] - measurements[:, :3])**2, axis=1))

    ax3.plot(frames, kf_pos_error, 'r-', linewidth=2, label='KF位置误差')
    ax3.plot(frames, ukf_pos_error, 'g-', linewidth=2, label='UKF位置误差')
    ax3.set_xlabel('帧数')
    ax3.set_ylabel('位置误差 (m)')
    ax3.set_title('位置估计误差')
    ax3.legend()
    ax3.grid(True)

    # 4. X方向位置对比
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(frames, measurements[:, 0], 'b.', alpha=0.5, label='观测值', markersize=3)
    ax4.plot(frames, kf_estimates[:, 0], 'r-', linewidth=2, label='KF估计')
    ax4.plot(frames, ukf_estimates[:, 0], 'g-', linewidth=2, label='UKF估计')
    ax4.set_xlabel('帧数')
    ax4.set_ylabel('X位置 (m)')
    ax4.set_title('X方向位置')
    ax4.legend()
    ax4.grid(True)

    # 5. 速度大小对比
    ax5 = fig.add_subplot(2, 3, 5)
    meas_speed = np.sqrt(np.sum(measurements[:, 3:6]**2, axis=1))
    kf_speed = np.sqrt(np.sum(kf_estimates[:, 3:6]**2, axis=1))
    ukf_speed = np.sqrt(np.sum(ukf_estimates[:, 3:6]**2, axis=1))

    ax5.plot(frames, meas_speed, 'b.', alpha=0.5, label='观测值', markersize=3)
    ax5.plot(frames, kf_speed, 'r-', linewidth=2, label='KF估计')
    ax5.plot(frames, ukf_speed, 'g-', linewidth=2, label='UKF估计')
    ax5.set_xlabel('帧数')
    ax5.set_ylabel('速度大小 (m/s)')
    ax5.set_title('速度估计')
    ax5.legend()
    ax5.grid(True)

    # 6. 速度误差对比
    ax6 = fig.add_subplot(2, 3, 6)
    kf_vel_error = np.sqrt(np.sum((kf_estimates[:, 3:6] - measurements[:, 3:6])**2, axis=1))
    ukf_vel_error = np.sqrt(np.sum((ukf_estimates[:, 3:6] - measurements[:, 3:6])**2, axis=1))

    ax6.plot(frames, kf_vel_error, 'r-', linewidth=2, label='KF速度误差')
    ax6.plot(frames, ukf_vel_error, 'g-', linewidth=2, label='UKF速度误差')
    ax6.set_xlabel('帧数')
    ax6.set_ylabel('速度误差 (m/s)')
    ax6.set_title('速度估计误差')
    ax6.legend()
    ax6.grid(True)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")

    plt.show()


def calculate_statistics(measurements, estimates, name=""):
    """
    计算并打印统计信息

    参数:
        measurements: 观测数据
        estimates: 估计结果
        name: 滤波器名称
    """
    # 位置误差
    pos_errors = np.sqrt(np.sum((estimates[:, :3] - measurements[:, :3])**2, axis=1))

    # 速度误差
    vel_errors = np.sqrt(np.sum((estimates[:, 3:6] - measurements[:, 3:6])**2, axis=1))

    print(f"\n{'='*50}")
    print(f"{name} 性能统计:")
    print(f"{'='*50}")
    print(f"位置误差 - 均值: {np.mean(pos_errors):.2f} m")
    print(f"位置误差 - 标准差: {np.std(pos_errors):.2f} m")
    print(f"位置误差 - 最大值: {np.max(pos_errors):.2f} m")
    print(f"位置误差 - RMSE: {np.sqrt(np.mean(pos_errors**2)):.2f} m")
    print(f"\n速度误差 - 均值: {np.mean(vel_errors):.2f} m/s")
    print(f"速度误差 - 标准差: {np.std(vel_errors):.2f} m/s")
    print(f"速度误差 - 最大值: {np.max(vel_errors):.2f} m/s")
    print(f"速度误差 - RMSE: {np.sqrt(np.mean(vel_errors**2)):.2f} m/s")


def plot_comparison(ship_meas, ship_kf, ship_ukf,
                   inter_meas, inter_ukf1, inter_ukf2):
    """
    绘制完整的对比结果

    参数:
        ship_meas: 舰船观测数据
        ship_kf: 舰船KF估计
        ship_ukf: 舰船UKF估计
        inter_meas: 拦截弹观测数据
        inter_ukf1: 拦截弹UKF估计1
        inter_ukf2: 拦截弹UKF估计2
    """
    # 绘制舰船跟踪结果
    plot_tracking_results(
        ship_meas, ship_kf, ship_ukf,
        title="舰船目标跟踪 - KF vs UKF",
        save_path="results/ship_tracking.png"
    )

    # 绘制拦截弹跟踪结果
    plot_tracking_results(
        inter_meas, inter_ukf1, inter_ukf2,
        title="拦截弹目标跟踪 - UKF对比",
        save_path="results/interceptor_tracking.png"
    )

    # 打印统计信息
    print("\n" + "="*60)
    print("舰船跟踪性能统计")
    print("="*60)
    calculate_statistics(ship_meas, ship_kf, "舰船-KF")
    calculate_statistics(ship_meas, ship_ukf, "舰船-UKF")

    print("\n" + "="*60)
    print("拦截弹跟踪性能统计")
    print("="*60)
    calculate_statistics(inter_meas, inter_ukf1, "拦截弹-UKF配置1")
    calculate_statistics(inter_meas, inter_ukf2, "拦截弹-UKF配置2")