# 使用示例

本文档提供更多使用示例和代码片段。

## 基础示例

### 示例1: 使用KF跟踪单个目标

```python
import numpy as np
from src.kalman_filter import KalmanFilter

# 创建卡尔曼滤波器
kf = KalmanFilter(dt=1.0)

# 模拟观测数据
observations = [
    [0, 0, 0, 10, 5, 0],      # [x, y, z, vx, vy, vz]
    [10, 5, 0, 10, 5, 0],
    [20, 10, 0, 10, 5, 0],
    # ... 更多观测
]

# 处理每个观测
estimates = []
for obs in observations:
    state = kf.process(obs)
    estimates.append(state)
    print(f"估计位置: ({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f})")
```

### 示例2: 使用UKF跟踪机动目标

```python
from src.unscented_kalman_filter import UnscentedKalmanFilter

# 创建UKF（调整参数以适应机动性）
ukf = UnscentedKalmanFilter(
    dt=1.0,
    alpha=0.01,    # 增大alpha以处理更大的非线性
    beta=2,
    kappa=1
)

# 处理观测
for obs in observations:
    state = ukf.process(obs)
    # 获取协方差（不确定性）
    P = ukf.get_covariance()
    print(f"位置不确定性: {np.sqrt(P[0,0]):.2f}m")
```

### 示例3: 处理漏检情况

```python
from src.kalman_filter import KalmanFilter

kf = KalmanFilter(dt=1.0)

# 观测数据，None表示未探测到
observations = [
    [0, 0, 0, 10, 5, 0],
    [10, 5, 0, 10, 5, 0],
    None,  # 第3帧未探测到
    None,  # 第4帧未探测到
    [40, 20, 0, 10, 5, 0],  # 第5帧重新探测到
]

for i, obs in enumerate(observations):
    if obs is not None:
        state = kf.process(obs)
        print(f"帧{i}: 更新 - 位置 ({state[0]:.1f}, {state[1]:.1f})")
    else:
        state = kf.process(None)  # 仅预测
        print(f"帧{i}: 预测 - 位置 ({state[0]:.1f}, {state[1]:.1f})")
```

## 高级示例

### 示例4: 自定义运动模型

```python
from src.unscented_kalman_filter import UnscentedKalmanFilter
import numpy as np

class CustomUKF(UnscentedKalmanFilter):
    """
    自定义UKF，使用恒定加速度模型
    状态: [x, y, z, vx, vy, vz, ax, ay, az]
    """
    
    def __init__(self, dt=1.0):
        super().__init__(dt)
        self.dim_x = 9  # 扩展到包含加速度
        
    def _state_transition(self, x):
        """恒定加速度模型"""
        dt = self.dt
        dt2 = dt * dt / 2
        
        x_new = np.zeros(9)
        # 位置更新
        x_new[0] = x[0] + x[3]*dt + x[6]*dt2
        x_new[1] = x[1] + x[4]*dt + x[7]*dt2
        x_new[2] = x[2] + x[5]*dt + x[8]*dt2
        # 速度更新
        x_new[3] = x[3] + x[6]*dt
        x_new[4] = x[4] + x[7]*dt
        x_new[5] = x[5] + x[8]*dt
        # 加速度保持不变
        x_new[6:9] = x[6:9]
        
        return x_new

# 使用
ukf = CustomUKF(dt=1.0)
```

### 示例5: 多目标跟踪

```python
from src.kalman_filter import KalmanFilter
import numpy as np

class MultiTargetTracker:
    """简单的多目标跟踪器"""
    
    def __init__(self, dt=1.0):
        self.dt = dt
        self.trackers = {}  # target_id -> KF
        self.next_id = 0
        
    def update(self, detections):
        """
        更新所有目标
        detections: list of (x, y, z, vx, vy, vz)
        """
        # 简化版：假设检测已关联到目标
        for target_id, detection in enumerate(detections):
            if target_id not in self.trackers:
                # 新目标
                self.trackers[target_id] = KalmanFilter(self.dt)
            
            # 更新跟踪
            state = self.trackers[target_id].process(detection)
        
        return self.get_all_states()
    
    def get_all_states(self):
        """获取所有目标的状态"""
        return {tid: kf.get_state() 
                for tid, kf in self.trackers.items()}

# 使用
tracker = MultiTargetTracker(dt=1.0)

# 假设每帧有多个目标
frame1_detections = [
    [0, 0, 0, 10, 5, 0],    # 目标0
    [100, 50, 0, -5, 10, 0] # 目标1
]

states = tracker.update(frame1_detections)
print(f"跟踪到 {len(states)} 个目标")
```

### 示例6: 实时绘图

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.kalman_filter import KalmanFilter
import numpy as np

class RealTimePlotter:
    """实时可视化跟踪结果"""
    
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        
        self.obs_scatter, = self.ax.plot([], [], 'b.', label='观测')
        self.est_line, = self.ax.plot([], [], 'r-', label='估计')
        
        self.observations = []
        self.estimates = []
        
        self.kf = KalmanFilter(dt=1.0)
        
    def update_plot(self, frame):
        # 模拟新观测
        t = frame * 0.1
        obs = [
            10*t + np.random.randn(),
            5*t + np.random.randn(),
            0, 10, 5, 0
        ]
        
        # 更新滤波器
        state = self.kf.process(obs)
        
        # 记录数据
        self.observations.append(obs[:2])
        self.estimates.append(state[:2])
        
        # 更新图形
        if len(self.observations) > 0:
            obs_arr = np.array(self.observations)
            est_arr = np.array(self.estimates)
            
            self.obs_scatter.set_data(obs_arr[:, 0], obs_arr[:, 1])
            self.est_line.set_data(est_arr[:, 0], est_arr[:, 1])
        
        return self.obs_scatter, self.est_line
    
    def animate(self):
        ani = FuncAnimation(
            self.fig, self.update_plot,
            frames=100, interval=100,
            blit=True
        )
        plt.legend()
        plt.show()

# 使用
plotter = RealTimePlotter()
plotter.animate()
```

### 示例7: 性能基准测试

```python
import time
import numpy as np
from src.kalman_filter import KalmanFilter
from src.unscented_kalman_filter import UnscentedKalmanFilter

def benchmark_filter(FilterClass, num_frames=1000):
    """测试滤波器性能"""
    filter = FilterClass(dt=1.0)
    
    # 生成测试数据
    observations = np.random.randn(num_frames, 6) * 10
    
    # 计时
    start_time = time.time()
    
    for obs in observations:
        filter.process(obs)
    
    elapsed = time.time() - start_time
    fps = num_frames / elapsed
    
    return elapsed, fps

# 运行基准测试
print("性能基准测试")
print("="*50)

kf_time, kf_fps = benchmark_filter(KalmanFilter)
print(f"KF: {kf_time:.3f}秒, {kf_fps:.1f} FPS")

ukf_time, ukf_fps = benchmark_filter(UnscentedKalmanFilter)
print(f"UKF: {ukf_time:.3f}秒, {ukf_fps:.1f} FPS")

print(f"\nKF比UKF快 {ukf_time/kf_time:.1f}x")
```

### 示例8: 保存和加载滤波器状态

```python
import pickle
from src.kalman_filter import KalmanFilter

# 创建并运行滤波器
kf = KalmanFilter(dt=1.0)
for obs in observations[:50]:  # 处理前50帧
    kf.process(obs)

# 保存状态
with open('kf_state.pkl', 'wb') as f:
    pickle.dump({
        'x': kf.x,
        'P': kf.P,
        'frame': 50
    }, f)

print("滤波器状态已保存")

# 稍后加载状态
with open('kf_state.pkl', 'rb') as f:
    state_dict = pickle.load(f)

# 创建新滤波器并恢复状态
kf_new = KalmanFilter(dt=1.0)
kf_new.x = state_dict['x']
kf_new.P = state_dict['P']
kf_new.initialized = True

# 从第51帧继续处理
for obs in observations[50:]:
    kf_new.process(obs)
```

### 示例9: 参数调优

```python
from src.unscented_kalman_filter import UnscentedKalmanFilter
import numpy as np

def evaluate_parameters(observations, ground_truth, 
                       alpha_range, beta_range):
    """
    评估不同参数组合的性能
    """
    results = []
    
    for alpha in alpha_range:
        for beta in beta_range:
            ukf = UnscentedKalmanFilter(
                dt=1.0, alpha=alpha, beta=beta
            )
            
            estimates = []
            for obs in observations:
                state = ukf.process(obs)
                estimates.append(state)
            
            # 计算RMSE
            estimates = np.array(estimates)
            rmse = np.sqrt(np.mean(
                (estimates[:, :3] - ground_truth[:, :3])**2
            ))
            
            results.append({
                'alpha': alpha,
                'beta': beta,
                'rmse': rmse
            })
    
    # 找到最佳参数
    best = min(results, key=lambda x: x['rmse'])
    print(f"最佳参数: alpha={best['alpha']}, beta={best['beta']}")
    print(f"RMSE: {best['rmse']:.2f}m")
    
    return results

# 使用
alpha_range = [0.001, 0.01, 0.1]
beta_range = [0, 2, 4]
results = evaluate_parameters(observations, ground_truth, 
                              alpha_range, beta_range)
```

### 示例10: 导出结果到不同格式

```python
import pandas as pd
import json
from src.kalman_filter import KalmanFilter

# 运行跟踪
kf = KalmanFilter(dt=1.0)
results = []

for i, obs in enumerate(observations):
    state = kf.process(obs)
    cov = kf.get_covariance()
    
    results.append({
        'frame': i,
        'x': state[0], 'y': state[1], 'z': state[2],
        'vx': state[3], 'vy': state[4], 'vz': state[5],
        'pos_uncertainty': np.sqrt(cov[0,0]),
        'vel_uncertainty': np.sqrt(cov[3,3])
    })

# 导出为CSV
df = pd.DataFrame(results)
df.to_csv('tracking_results.csv', index=False)

# 导出为JSON
with open('tracking_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 导出为Excel
df.to_excel('tracking_results.xlsx', index=False)

print("结果已导出为多种格式")
```

## 集成示例

### 与ROS集成（机器人操作系统）

```python
#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from src.kalman_filter import KalmanFilter
import numpy as np

class ROSTracker:
    def __init__(self):
        rospy.init_node('kf_tracker')
        
        self.kf = KalmanFilter(dt=0.1)  # 10Hz
        
        # 订阅观测
        self.sub = rospy.Subscriber(
            '/object/pose', PoseStamped,
            self.callback
        )
        
        # 发布估计
        self.pub = rospy.Publisher(
            '/object/filtered_pose', PoseStamped,
            queue_size=10
        )
    
    def callback(self, msg):
        # 转换ROS消息为观测
        obs = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            0, 0, 0  # 假设速度从差分计算
        ])
        
        # 更新滤波器
        state = self.kf.process(obs)
        
        # 发布估计结果
        est_msg = PoseStamped()
        est_msg.header = msg.header
        est_msg.pose.position.x = state[0]
        est_msg.pose.position.y = state[1]
        est_msg.pose.position.z = state[2]
        
        self.pub.publish(est_msg)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    tracker = ROSTracker()
    tracker.run()
```

## 小贴士

1. **选择合适的时间步长**: `dt` 应该匹配你的采样率
2. **调整过程噪声**: 如果目标运动不规则，增大 `Q`
3. **调整观测噪声**: 如果传感器精度高，减小 `R`
4. **使用UKF处理机动**: 对于高机动性目标，UKF通常表现更好
5. **处理漏检**: 在连续多帧漏检时，考虑增加不确定性
6. **多传感器融合**: 可以扩展为处理来自多个传感器的观测

更多示例和文档，请访问项目Wiki或提交Issue询问！