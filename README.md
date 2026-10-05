# MNIST 手写数字识别

TensorFlow/Keras CNN、Flask 和 Canvas 实现的单个手写数字识别应用。
黑底白字输入经过裁剪、等比例缩放到最长边 20 像素及重心居中，输出为 28×28。
支持鼠标、触摸、触控笔；请求期间暂停绘制，15 秒超时后恢复操作。

## 安装和运行

支持 Python 3.11–3.13。使用经过验证、包含 SHA-256 的依赖锁文件：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python app.py
```

打开 <http://127.0.0.1:5000>。安装 wheel 后可从任意目录运行 `mnist-web`。
生产服务使用 `waitress-serve --host=127.0.0.1 --port=5000 mnist_web.app:app`。
开发调试器默认关闭，仅在明确需要时设置 `FLASK_DEBUG=1`。

## 固定模型与配置

默认加载随 Git、sdist 和 wheel 一起发布的 `mnist_web/assets/model.keras`。
它的 `model.manifest.json` 固定模型编号、SHA-256、预处理版本和类别顺序。
加载后先校验哈希，再执行预热并验证输出；日志记录实际模型身份。
根目录遗留的 `model.h5`、`model.keras` 不会自动替换发布模型。

使用自行训练的模型时显式配置路径，并保留训练生成的同名 manifest：

```powershell
$env:MNIST_MODEL_PATH = (Resolve-Path artifacts/model.keras).Path
python app.py
```

可额外设置 `MNIST_MODEL_SHA256` 固定预期哈希，设置 `MNIST_CALIBRATION_PATH` 加载
独立画板验证集生成的阈值文件。阈值必须匹配模型哈希和预处理版本。
未提供有效验证证据时，`uncertain` 为 `null`，界面展示“模型分数”，不采用任意的 0.6 阈值。
高分不保证输入是一个数字，也不代表实际正确率。

## API 和健康检查

- `GET /health`：存活检查，不需要模型可用。
- `GET /ready`：确认模型能够加载和推理，成功返回身份信息，失败返回 JSON 503。
- `POST /predict`：接收 `{"image":"data:image/png;base64,..."}`，返回
  `prediction`、`confidence`、`top3`、`uncertain`、`confidence_threshold` 和 `model`。

无效请求、空画布、非 PNG、超过 2048 像素的图片返回 JSON 400；
请求体超过 2 MB 返回 JSON 413；模型加载或推理失败返回 JSON 503 并记录服务端日志。
透明 PNG 合成到黑色背景后再预处理，完全透明的隐藏笔迹不会被当作输入。

## 训练和复现

```powershell
python model.py --output artifacts/model.keras --epochs 20 --batch-size 128 --seed 42
```

默认 `--preprocessing drawing` 对训练、验证和测试都应用与服务端一致的裁剪/居中流程。
可用 `--preprocessing normalized` 做仅归一化的对照实验。数据增强使用黑色常量填充；
划分按类别分层，随机种子统一设置，默认启用确定性操作。
早停、最佳检查点和学习率调整均监控 `val_loss`；最终评测读取保存的最佳模型。

生成 `.keras`、`.manifest.json`、`.metadata.json` 和 `.split.npz`。
元数据记录最佳 epoch、环境版本、硬件、数据与划分哈希、预处理模式和测试指标。
确定性复现应使用相同依赖、设备和环境；跨硬件不能保证逐位相同。
训练输出保存在 `artifacts/`，不会隐式覆盖发布模型。

## 画板数据采集与独立评测

打开 <http://127.0.0.1:5000/collect>，填写匿名书写者编号、选择真实标签、保存并导出 JSON。
样本只留在页面内存，离开前需导出；没有自动上传或保存到服务器。
建议覆盖 0–9，各类自然粗细、大小、位置、倾斜和易混淆写法；乱画/多个数字标记为 -1。
不同书写者应分别放入验证集和测试集，保持匿名编号稳定。

标准和合成基准（完整 MNIST 测试集、各类别指标、混淆矩阵和可靠性诊断）：

```powershell
python -m mnist_web.evaluation --output reports/local/evaluation.json
```

真实画板评测与阈值选择：

```powershell
python -m mnist_web.evaluation --canvas-validation datasets/validation.json --canvas-test datasets/test.json --calibration-output artifacts/calibration.json --output reports/local/human-evaluation.json
```

阈值在验证集上选择，在独立测试集上报告接受准确率和覆盖率。
默认至少保留 30 个验证样本、达到 95% 的经验接受准确率；证据不足时不启用阈值。
目标是经验筛选规则，不是统计保证或概率校准。评测拒绝图像重复与验证/测试书写者重叠。
合成画板结果始终独立标注，不能替代真实用户准确率。

本次完整评测：发布模型 MNIST 准确率 **99.21%**。统一预处理的新候选达到 **99.30%**
原始 MNIST 准确率，但在加粗倾斜合成样本上有退步，因此暂保留已有模型为默认。
完整对照、性能数字和局限见 [优化记录](reports/optimization.md)。

## 检查、安装包与模型发布产物

```powershell
python -m ruff check .
python -m pytest
node --test tests/frontend.test.cjs
python -m mnist_web.benchmark --output reports/local/benchmark.json
python -m build --no-isolation
python -m mnist_web.release --evaluation reports/evaluation.json --output dist/model-bundle
```

wheel 包含模板、静态资源、固定模型和 manifest，支持离开源码目录运行。
发布工具将模型、manifest、元数据、对应评测和依赖锁文件复制到一个空目录，生成 `SHA256SUMS`。
GitHub Actions 对 Windows/Linux、Python 3.11/3.13 执行检查和安装验证；
手动运行 `Model and wheel release bundle` 工作流会生成可下载的发布附件。
工作流生成产物，公开发布 Release 由维护者决定。

依赖锁更新方式（工具使用 uv，约束文件记录本次验证的直接依赖）：

```powershell
uv pip compile pyproject.toml --extra dev --universal --python-version 3.11 --generate-hashes --constraint constraints-tested.txt --output-file requirements.lock
```

## 结构

```text
app.py / model.py / preprocessing.py    原有命令的兼容入口
mnist_web/app.py                       推理接口、存活与就绪检查
mnist_web/preprocessing.py             训练与推理共用预处理
mnist_web/training.py                  可复现训练
mnist_web/evaluation.py                标准、合成、人类画板评测
mnist_web/artifacts.py / release.py    模型校验与发布打包
mnist_web/benchmark.py                 推理与接口性能基准
mnist_web/assets/                      固定发布模型和来源记录
mnist_web/templates/ / static/         随包发布的网页资源
tests/                                Python 集成测试和前端行为测试
reports/                              本次评测与优化记录
```

项目使用 MIT License，见 [LICENSE](LICENSE)。
