# 项目设置详细指南

本文档提供项目的详细设置步骤和使用说明。

## 目录结构创建

首先创建完整的项目目录结构：

```bash
mkdir target-tracking
cd target-tracking

# 创建目录
mkdir src data results

# 创建空文件（Unix/Linux/Mac）
touch src/__init__.py
touch src/generate_data.py
touch src/kalman_filter.py
touch src/unscented_kalman_filter.py
touch src/visualization.py
touch main.py
touch README.md
touch environment.yml
touch requirements.txt
touch .gitignore

# Windows用户使用
# type nul > filename.txt
```

## 文件说明

### 核心代码文件

1. **src/__init__.py** - Python包初始化文件
2. **src/generate_data.py** - 生成轨迹数据
3. **src/kalman_filter.py** - 卡尔曼滤波器实现
4. **src/unscented_kalman_filter.py** - 无迹卡尔曼滤波器实现
5. **src/visualization.py** - 可视化函数
6. **main.py** - 主程序

### 配置文件

1. **environment.yml** - Conda环境配置
2. **requirements.txt** - pip依赖列表
3. **.gitignore** - Git忽略规则

### 文档文件

1. **README.md** - 项目说明
2. **SETUP.md** - 设置指南（本文件）

## 环境安装步骤

### 选项A: 使用Conda（推荐）

#### 1. 安装Anaconda/Miniconda

如果还没有安装Conda，先下载并安装：
- Anaconda: https://www.anaconda.com/products/distribution
- Miniconda（更轻量）: https://docs.conda.io/en/latest/miniconda.html

#### 2. 创建环境

```bash
# 从environment.yml创建环境
conda env create -f environment.yml

# 如果要指定不同的环境名
conda env create -f environment.yml -n my-tracking-env
```

#### 3. 激活环境

```bash
# 激活环境
conda activate target-tracking

# 验证安装
python --version  # 应该显示Python 3.9.x
python -c "import numpy; print(numpy.__version__)"
```

#### 4. 环境管理命令

```bash
# 查看所有环境
conda env list

# 停用环境
conda deactivate

# 删除环境
conda env remove -n target-tracking

# 更新环境
conda env update -f environment.yml
```

### 选项B: 使用pip和venv

#### 1. 创建虚拟环境

```bash
# Python 3.7+
python -m venv venv

# 或使用python3
python3 -m venv venv
```

#### 2. 激活虚拟环境

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

如果PowerShell报错"无法加载文件"，先运行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 3. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. 验证安装

```bash
pip list
python -c "import numpy, pandas, matplotlib; print('All packages imported successfully!')"
```

## 运行程序

### 第一次运行

```bash
# 确保在项目根目录
cd target-tracking

# 激活环境
conda activate target-tracking  # 或 source venv/bin/activate

# 运行主程序
python main.py
```

### 仅生成数据

```bash
python src/generate_data.py
```

### 查看生成的数据

```bash
# 使用pandas查看
python -c "import pandas as pd; print(pd.read_csv('data/ship_trajectory.csv').head())"

# 或使用任何文本编辑器/Excel打开CSV文件
```

## 结果输出

运行完成后，检查输出：

```bash
# 查看数据文件
ls -lh data/

# 查看结果图片
ls -lh results/

# 在图形界面打开图片
# Mac
open results/ship_tracking.png

# Linux
xdg-open results/ship_tracking.png

# Windows
start results/ship_tracking.png
```

## 常见问题排查

### 问题1: ModuleNotFoundError

```bash
# 确认环境已激活
which python  # Linux/Mac
where python  # Windows

# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

### 问题2: 中文显示乱码

如果图表中文显示为方框，修改 `visualization.py`：

```python
# 添加字体配置
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
# 或
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows
# 或
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # Linux
```

### 问题3: 图形不显示

```bash
# 如果运行在远程服务器或没有图形界面
# 修改visualization.py中的plt.show()为plt.savefig()

# 或设置matplotlib后端
export MPLBACKEND=Agg  # Linux/Mac
set MPLBACKEND=Agg  # Windows
```

### 问题4: 权限错误

```bash
# Linux/Mac: 确保有执行权限
chmod +x main.py

# 或使用python明确调用
python main.py
```

### 问题5: NumPy/SciPy安装失败

```bash
# 使用conda安装（更稳定）
conda install numpy scipy matplotlib pandas

# 或安装预编译包
pip install --only-binary :all: numpy scipy
```

## 性能优化建议

### 1. 使用多进程处理多个目标

```python
from multiprocessing import Pool

def process_target(target_data):
    # 处理单个目标
    pass

with Pool(4) as p:
    results = p.map(process_target, targets_list)
```

### 2. 使用Numba加速

```bash
pip install numba
```

```python
from numba import jit

@jit(nopython=True)
def fast_matrix_multiply(A, B):
    return A @ B
```

### 3. 批处理数据

```python
# 一次处理多个帧
batch_size = 10
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    process_batch(batch)
```

## 开发建议

### 代码格式化

```bash
# 安装工具
pip install black flake8

# 格式化代码
black src/ main.py

# 检查代码风格
flake8 src/ main.py
```

### 单元测试

创建 `tests/` 目录：

```bash
mkdir tests
touch tests/__init__.py
touch tests/test_kalman_filter.py
```

```python
# tests/test_kalman_filter.py
import unittest
from src.kalman_filter import KalmanFilter

class TestKalmanFilter(unittest.TestCase):
    def test_initialization(self):
        kf = KalmanFilter()
        self.assertEqual(kf.dim_x, 6)
    
    def test_predict(self):
        kf = KalmanFilter()
        kf.initialize(np.zeros(6))
        state = kf.predict()
        self.assertEqual(len(state), 6)

if __name__ == '__main__':
    unittest.main()
```

运行测试：
```bash
python -m unittest discover tests
```

## Git使用

### 初始化仓库

```bash
git init
git add .
git commit -m "Initial commit: KF/UKF tracking system"
```

### 创建远程仓库

```bash
# 在GitHub/GitLab创建仓库后
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

### 常用命令

```bash
# 查看状态
git status

# 提交更改
git add .
git commit -m "Description of changes"
git push

# 查看历史
git log --oneline

# 创建分支
git checkout -b feature/new-feature
```

## 部署到服务器

### 使用Docker（可选）

创建 `Dockerfile`：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

构建和运行：

```bash
docker build -t target-tracking .
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results target-tracking
```

## 扩展功能

### 添加实时可视化

```bash
pip install dash plotly
```

### 添加数据库支持

```bash
pip install sqlalchemy
```

### 添加Web界面

```bash
pip install flask
```

## 支持与反馈

如遇到问题：

1. 检查本文档的常见问题部分
2. 查看 `README.md` 中的FAQ
3. 在GitHub Issues中搜索类似问题
4. 创建新的Issue描述问题

---

祝使用愉快！🚀