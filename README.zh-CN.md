# MNIST · 手写数字识别

[English](README.md) · [简体中文](README.zh-CN.md)

基于 TensorFlow/Keras 与 Flask 的手写数字识别应用，提供浏览器画板、预训练 CNN，以及训练、评测和模型打包工具。

在浏览器中书写一个数字，即可查看预测结果及分数最高的三个候选。仓库包含本地运行所需的模型与网页资源，同时提供带版本的预处理流程、评测报告、自动化测试和交付工作流。

| 项目概览 | 说明 |
| --- | --- |
| 识别任务 | 单个手写数字分类，类别为 0–9 |
| 模型 | 两个卷积模块与十分类 Softmax 输出 |
| 已记录的 MNIST 准确率 | 完整 10,000 张测试图像上为 **99.23%**（经服务端预处理为 99.24%） |
| 运行环境 | Python 3.11–3.13；TensorFlow/Keras；Flask |
| 交付形式 | Python wheel、源码分发包与带校验和的模型包 |
| 许可证 | [MIT](LICENSE) |

## 快速开始

使用 Python 3.11–3.13。应用默认加载仓库内置模型，无需先训练。

```bash
git clone https://github.com/MapleSugarMochi/MNIST.git
cd MNIST
python -m venv .venv
```

在 Windows PowerShell 中激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

在 Linux/macOS 中激活环境：

```bash
source .venv/bin/activate
```

安装带哈希校验的锁定依赖与应用：

```bash
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python app.py
```

打开 [http://127.0.0.1:5000](http://127.0.0.1:5000)，在画板中写一个数字，点击“识别”。界面目前使用中文，支持鼠标、触摸和触控笔。

安装后也可用 `mnist-web` 启动应用。wheel 包含模型、模板和静态资源，支持离开源码目录运行。当前 CI 覆盖 Windows 与 Linux，macOS 尚未纳入 CI 矩阵。

## 核心能力与推理流程

- **交互式识别：** 候选排名与模型分数。请求期间暂停绘制，15 秒超时后恢复操作。
- **统一输入处理：** 裁剪可见笔迹、保持宽高比缩放、按像素重心居中并归一化。
- **模型身份校验：** 校验模型 SHA-256，预热推理，检查输出，并返回实际加载的模型身份。
- **明确的服务行为：** 分离存活与就绪检查，限制图片输入，使用 JSON 返回错误。
- **评测工具：** 支持 MNIST 指标、合成画板变体及独立的真实画板评测。
- **可追溯产物：** 模型 manifest、训练元数据、带哈希的依赖锁文件及带校验和的发布包。

```mermaid
flowchart LR
    A[浏览器画板] --> B[PNG 输入校验]
    B --> C[裁剪可见笔迹]
    C --> D[缩放至最长边 20 像素]
    D --> E[重心居中与归一化：28 × 28]
    E --> F[CNN 推理]
    F --> G[预测结果、前三候选、模型身份]
```

画板采用黑底白字。透明 PNG 在处理前合成到黑色背景。模型输入为形状 `(1, 28, 28, 1)` 的 float32 张量，取值范围为 `[0, 1]`。

## 评测结果

仓库内的[评测报告](reports/evaluation.json)对应模型 `mnist-drawing-seed42-20261010T103405Z`。[评测记录](reports/optimization.md)标注的评测日期为 2026 年 10 月 10 日。下表每项均使用完整 MNIST 测试集的 10,000 张图像。

| 输入条件 | 准确率 |
| --- | ---: |
| MNIST，仅归一化 | **99.23%** |
| MNIST，裁剪与重心居中 | **99.24%** |
| 合成画板，普通大小 | 99.24% |
| 合成画板，小字与偏移 | 99.26% |
| 合成画板，加粗与 12° 倾斜 | 98.58% |
| 合成画板，细笔迹 | 99.18% |

合成画板由 MNIST 图像绘制到黑底 280 × 280 画布，再经过服务端预处理。这些结果衡量受控变换下的表现；独立的真实用户画板数据尚未采集。报告还包含各类别的 precision、recall、F1、混淆矩阵及分数可靠性诊断。

重新运行标准与合成评测，将结果写入本地目录：

```bash
python -m mnist_web.evaluation --output reports/local/evaluation.json
```

需要时，Keras 会下载 MNIST 数据集。评测其他模型时，添加 `--model artifacts/model.keras`。

### 性能测量

已记录的[性能基准](reports/benchmark.json)使用 2 vCPU 的 Linux x86-64 环境、Python 3.13.16 和 TensorFlow 2.20.0，在预热后测量 30 次调用。

| 操作 | 中位数 | p95 |
| --- | ---: | ---: |
| 直接调用模型 | 6.38 ms | 10.36 ms |
| Flask 测试客户端顺序请求 | 7.49 ms | 8.82 ms |
| Flask 测试客户端，4 个并发工作线程 | 35.06 ms | 45.40 ms |

以上为本机进程内测量，不包含 HTTP 网络与 WSGI 开销。评估部署性能时，应在目标环境中单独测量。

```bash
python -m mnist_web.benchmark --output reports/local/benchmark.json
```

## API

| 接口 | 用途 | 成功响应 |
| --- | --- | --- |
| `GET /health` | 检查应用存活，不要求模型可用 | HTTP 200 |
| `GET /ready` | 加载、校验并预热模型，返回模型身份 | HTTP 200 |
| `POST /predict` | 识别一张 PNG 画板图片 | HTTP 200 |
| `GET /collect` | 打开带标签的样本采集界面 | HTML 页面 |

向 `POST /predict` 提交包含 PNG Data URL 的 JSON：

```json
{
  "image": "data:image/png;base64,..."
}
```

| 响应字段 | 含义 |
| --- | --- |
| `prediction` | 分数最高的数字，范围为 0–9 |
| `confidence` | 该类别的 Softmax 分数 |
| `top3` | 按分数排序的三个候选，每项含 `digit` 与 `confidence` |
| `uncertain` | 分数是否低于配置阈值；未启用阈值时为 `null` |
| `confidence_threshold` | 当前阈值，或 `null` |
| `model` | 已加载模型的身份，包含 ID、SHA-256 与预处理版本 |

模型分数不代表正确率保证。未加载匹配的真实画板验证阈值文件时，`uncertain` 与 `confidence_threshold` 均为 `null`。

| 错误情况 | HTTP 状态码 |
| --- | ---: |
| 无效 JSON、缺少图片、无效或非 PNG 输入、空画板、图片边长超过 2048 像素 | 400 |
| 请求体超过 2 MiB | 413 |
| 模型加载或推理失败、就绪检查失败 | 503 |

错误以 JSON 返回。模型故障会记录到服务端日志。

## 模型与运行配置

默认模型为 [`mnist_web/assets/model.keras`](mnist_web/assets/model.keras)。对应的 [manifest](mnist_web/assets/model.manifest.json)记录模型 ID、SHA-256、输入形状、类别顺序及预处理版本。

| 环境变量 | 行为 |
| --- | --- |
| `MNIST_MODEL_PATH` | 加载自定义模型；需在其旁保留匹配的 `.manifest.json` |
| `MNIST_MODEL_SHA256` | 额外指定预期模型哈希 |
| `MNIST_CALIBRATION_PATH` | 加载与模型哈希及预处理版本匹配的真实画板验证阈值文件 |
| `PORT` | `python app.py` / `mnist-web` 的监听端口，默认 `5000` |
| `FLASK_DEBUG` | 仅设为 `1` 时启用开发调试器，默认关闭 |

训练自定义模型后，在 PowerShell 中配置：

```powershell
$env:MNIST_MODEL_PATH = (Resolve-Path artifacts/model.keras).Path
python app.py
```

在 POSIX shell 中配置：

```bash
export MNIST_MODEL_PATH="$PWD/artifacts/model.keras"
python app.py
```

使用 Waitress 提供服务：

```bash
waitress-serve --host=127.0.0.1 --port=5000 mnist_web.app:app
```

## 训练与复现

```bash
python model.py --output artifacts/model.keras --epochs 20 --batch-size 128 --seed 42
```

当前流程按类别分层划分 90% 训练集与 10% 验证集，统一设置随机种子，并默认启用 TensorFlow 确定性操作。默认 `--preprocessing drawing` 对训练、验证和测试图像应用服务端的裁剪与居中流程。可用 `--preprocessing normalized` 进行仅归一化的对照实验。

网络结构为 `Conv2D(32) → MaxPooling2D → Conv2D(64) → MaxPooling2D → Flatten → Dense(64) → Dense(10, Softmax)`。训练加入平移、旋转和缩放增强，使用黑色常量填充；优化器为 Adam，损失函数为稀疏分类交叉熵。早停、最佳检查点与学习率调整均监控 `val_loss`，评测重新加载保存的最佳检查点。

| 输出文件 | 内容 |
| --- | --- |
| `model.keras` | 保存的最佳模型 |
| `model.manifest.json` | 模型哈希、身份与输入输出约定 |
| `model.metadata.json` | 随机种子、环境、硬件、训练历史、最佳 epoch、预处理、数据/划分哈希与测试指标 |
| `model.split.npz` | 训练集与验证集索引 |

默认输出目录为 `artifacts/`。复现依赖相同的依赖版本、硬件和运行环境，跨设备不保证逐位一致。

内置模型由当前流程使用上述命令训练得到（默认 `drawing` 预处理，种子 42）。训练在第 16 轮早停，恢复 `val_loss` 最低的第 13 轮权重。其[元数据](mnist_web/assets/model.metadata.json)记录了运行环境、数据集与划分哈希、训练历史和测试指标。在同一环境中重复训练得到逐位相同的权重；由于 `.keras` 归档内含保存时间戳，两次产出的文件哈希不同。

## 真实画板评测

打开 [http://127.0.0.1:5000/collect](http://127.0.0.1:5000/collect)，为笔迹填写真实标签与匿名书写者编号，保存后导出 JSON。样本在导出前仅保存在当前页面内存中，离开页面前应先导出；采集过程不会自动上传样本到服务器。

样本应覆盖 0–9 的不同写法、大小、位置、笔迹粗细和倾斜。乱画或多个数字标记为 `-1`。保持书写者编号稳定，并将不同书写者分配到验证集和测试集。评测器拒绝重复图像与两份语料之间的书写者重叠。

导出两份数据集后运行：

```bash
python -m mnist_web.evaluation --canvas-validation datasets/validation.json --canvas-test datasets/test.json --calibration-output artifacts/calibration.json --output reports/local/human-evaluation.json
```

阈值在验证集上选择，再在独立测试集上报告接受准确率、覆盖率和非数字误接受数。默认要求至少接受 30 个验证样本，并达到 95% 的经验接受准确率；证据不足时不启用阈值。这是一项经验筛选规则，不构成概率校准或统计保证。

使用生成的阈值时，将 `MNIST_CALIBRATION_PATH` 设为该文件路径，并重启服务。

## 开发检查与交付

安装锁定依赖后，在仓库根目录运行：

```bash
python -m ruff check .
python -m pytest
node --test tests/frontend.test.cjs
python -m build --no-isolation
python -m mnist_web.release --evaluation reports/evaluation.json --output dist/model-bundle
```

前端测试需要 Node.js，CI 使用 Node.js 22。测试覆盖预处理、API 错误、模型身份与真实模型推理、训练产物、阈值选择、发布完整性及画板与请求行为。

[CI 工作流](.github/workflows/ci.yml)配置了 Windows/Linux 与 Python 3.11/3.13 的组合，执行代码检查、Python 与前端测试、分发包构建，以及离开源码目录后的 wheel 安装检查。

发布命令要求输出目录为空，且评测报告与模型哈希一致。它打包模型、manifest、评测报告及可用的元数据和依赖锁文件，并生成 `SHA256SUMS`。手动触发的 [Model and wheel release bundle 工作流](.github/workflows/release.yml)会评测模型并上传分发产物；公开 GitHub Release 由维护者发布。

使用 `uv` 更新依赖锁文件：

```bash
uv pip compile pyproject.toml --extra dev --universal --python-version 3.11 --generate-hashes --constraint constraints-tested.txt --output-file requirements.lock
```

## 项目结构

```text
app.py / model.py / preprocessing.py   兼容入口
mnist_web/
  app.py                              Flask 接口与推理服务
  preprocessing.py                    共用图像预处理
  training.py                         训练与来源记录
  evaluation.py                       MNIST、合成与真实画板评测
  benchmark.py                        推理与 API 性能测量
  artifacts.py / release.py            模型校验与发布打包
  assets/                             内置模型、manifest 与元数据
  templates/ / static/                 随包发布的浏览器界面
tests/                                Python 与前端测试
reports/                              已记录的指标与评测说明
scripts/check_installed.py            安装包验证
.github/workflows/                     CI 与产物工作流
```

## 适用范围与许可证

本项目适用于单个手写数字的实验、学习和本地推理。真实笔迹可能与 MNIST 存在分布差异，高分也不能证明输入一定是数字。多位数字识别、通用 OCR 以及任意非数字输入的可靠拒绝，均超出当前已验证的范围。

项目采用 [MIT License](LICENSE)。
