# 目标跟踪系统 - KF/UKF实现

基于卡尔曼滤波器(KF)和无迹卡尔曼滤波器(UKF)的目标跟踪系统，用于跟踪舰船和拦截弹的运动轨迹。

## 项目特性

- 🎯 **数据生成**: 自动生成舰船和拦截弹的运动轨迹数据
- 📊 **KF实现**: 标准卡尔曼滤波器，适用于线性系统
- 🔮 **UKF实现**: 无迹卡尔曼滤波器，更适合非线性和机动目标
- 📈 **可视化**: 3D轨迹、误差分析、性能对比
- 🚫 **漏检处理**: 支持目标探测缺失时的状态预测

## 项目结构

```
target-tracking/
├── README.md                 # 项目说明文档
├── environment.yml           # Conda环境配置
├── requirements.txt          # Python依赖包
├── .gitignore               # Git忽略文件配置
├── main.py                  # 主程序入口
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── generate_data.py     # 数据生成模块
│   ├── kalman_filter.py     # 卡尔曼滤波器实现
│   ├── unscented_kalman_filter.py  # UKF实现
│   └── visualization.py     # 可视化模块
├── data/                    # 数据存储目录
│   ├── ship_trajectory.csv
│   └── interceptor_trajectory.csv
└── results/                 # 结果输出目录
    ├── ship_tracking.png
    └── interceptor_tracking.png
```

## 环境配置

### 方法1: 使用Conda（推荐）

```bash
# 克隆项目
git clone <your-repo-url>
cd target-tracking

# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate target-tracking
```

### 方法2: 使用pip

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 生成数据

```bash
python src/generate_data.py
```

这将在 `data/` 目录下生成两个CSV文件：

- `ship_trajectory.csv`: 舰船运动轨迹（100帧）
- `interceptor_trajectory.csv`: 拦截弹运动轨迹（100帧）

数据格式：

| 列名      | 说明                           |
| --------- | ------------------------------ |
| frameID   | 帧ID                           |
| target_x  | X坐标 (米)                     |
| target_y  | Y坐标 (米)                     |
| target_z  | Z坐标 (米)                     |
| target_Vx | X方向速度 (米/秒)              |
| target_Vy | Y方向速度 (米/秒)              |
| target_Vz | Z方向速度 (米/秒)              |
| detected  | 探测标志 (1:已探测, -1:未探测) |

### 2. 运行跟踪算法

```bash
python main.py
```

这将：

1. 自动生成数据（如果不存在）
2. 使用KF跟踪舰船
3. 使用UKF跟踪舰船
4. 使用UKF跟踪拦截弹（两种参数配置）
5. 生成可视化结果并保存到 `results/` 目录

## 算法说明

### 卡尔曼滤波器 (KF)

**适用场景**: 线性系统、高斯噪声

**核心方程**:

- 预测: `x̂(k|k-1) = F·x̂(k-1|k-1)`
- 更新: `x̂(k|k) = x̂(k|k-1) + K·(z(k) - H·x̂(k|k-1))`

**优点**:

- 计算效率高
- 实现简单
- 对线性系统最优

**缺点**:

- 仅适用于线性系统
- 对非线性系统效果较差

### 无迹卡尔曼滤波器 (UKF)

**适用场景**: 非线性系统、机动目标

**核心思想**:

- 使用无迹变换通过一组sigma点捕捉状态分布
- 将sigma点通过非线性函数传播
- 比扩展卡尔曼滤波器(EKF)更准确

**优点**:

- 能处理强非线性
- 无需计算雅可比矩阵
- 精度比EKF高（二阶精度）

**参数说明**:

- `alpha`: sigma点分布参数（通常0.001-1）
- `beta`: 先验知识参数（高斯分布用2）
- `kappa`: 次要缩放参数（通常0或3-n）

## 性能对比

运行程序后，会在终端输出详细的性能统计：

```
========================================
舰船-KF 性能统计:
========================================
位置误差 - RMSE: X.XX m
速度误差 - RMSE: X.XX m/s

========================================
舰船-UKF 性能统计:
========================================
位置误差 - RMSE: X.XX m
速度误差 - RMSE: X.XX m/s
```

通常情况下：

- 对于机动目标，UKF的位置和速度估计误差更小
- KF在计算效率上更优
- 当探测缺失时，两者都能进行有效预测

## 可视化结果

程序会生成两个PNG图表，每个包含6个子图：

1. **3D轨迹对比**: 观测值、KF估计、UKF估计的三维轨迹
2. **X-Y平面投影**: 俯视图轨迹对比
3. **位置误差**: KF和UKF的位置估计误差对比
4. **X方向位置**: 时间序列上的X坐标对比
5. **速度估计**: 速度大小的时间序列对比
6. **速度误差**: 速度估计误差对比

## 高级用法

### 调整滤波器参数

编辑 `main.py` 中的参数：

```python
# KF参数在 kalman_filter.py 中调整
# 过程噪声
q = 0.1  # 增大可增强对突变的响应

# UKF参数
alpha = 0.001  # 增大可增加sigma点分布范围
beta = 2       # 高斯分布最优值
kappa = 0      # 调整可改变sigma点权重
```

### 自定义数据生成

编辑 `src/generate_data.py` 中的参数：

```python
# 修改初始速度
vx0, vy0, vz0 = 50.0, 30.0, 0.0

# 修改加速度
ax, ay, az = 0.5, -0.3, 0.0

# 修改探测概率
detected = 1 if np.random.rand() > 0.1 else -1  # 90%探测率
```

### 单独运行模块

```bash
# 仅生成数据
python src/generate_data.py

# 使用自定义数据
python -c "from main import run_kf_tracking; ..."
```

## 依赖说明

主要依赖包：

- `numpy>=1.21.0`: 数值计算
- `pandas>=1.3.0`: 数据处理
- `matplotlib>=3.4.0`: 可视化
- `scipy>=1.7.0`: 科学计算

完整依赖列表见 `requirements.txt`

## 常见问题

### Q: 为什么有些帧显示未探测到？

A: 模拟真实场景中的探测不确定性。当目标未被探测到时，滤波器会使用运动模型进行状态预测。

### Q: KF和UKF哪个更好？

A: 对于线性匀速运动，KF效率更高；对于机动目标，UKF精度更高。实际应用中可根据目标特性选择。

### Q: 如何提高跟踪精度？

A:

1. 调整过程噪声协方差Q
2. 调整观测噪声协方差R
3. 对UKF，调整alpha参数
4. 使用更准确的运动模型

### Q: 能处理多目标跟踪吗？

A: 当前实现是单目标。多目标需要添加数据关联算法（如JPDA、MHT）。

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

MIT License

## 联系方式

如有问题，请提交Issue或联系项目维护者。

## 参考资料

1. Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
2. Julier, S. J., & Uhlmann, J. K. (1997). "New extension of the Kalman filter to nonlinear systems"
3. Wan, E. A., & Van Der Merwe, R. (2000). "The unscented Kalman filter for nonlinear estimation"

## 更新日志

### v1.0.0 (2024-11-07)

- 初始版本发布
- 实现KF和UKF算法
- 添加数据生成和可视化功能
- 支持漏检场景处理
