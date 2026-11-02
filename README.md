# NumPy 全连接网络：MNIST 十分类

本项目按 `command.md` 实现 `784 → 128 → 64 → 10` 网络。两个隐藏层均使用 sigmoid，输出层使用 softmax，损失为多分类交叉熵；使用手写反向传播和普通小批量梯度下降。只依赖 NumPy 和 Matplotlib，不使用自动求导或神经网络框架。未改动相邻逻辑回归项目。本文件是代码与运行说明，不是实验报告。

## 代码组织

| 文件 | 职责 |
| --- | --- |
| `src/train.py` | 神经网络核心：参数初始化、激活函数、前向传播、交叉熵、反向传播、参数更新、预测评估及训练循环 |
| `src/experiment_utils.py` | 实验辅助：实现验证、数值检查、曲线与图片生成、具体样本说明、日志、CSV/JSON/模型参数保存，以及命令行入口 |
| `src/load_data.py` | MNIST 查找与读取、展平、归一化、one-hot 标签生成 |

学习网络时，先阅读 `train.py`。其中 `train_epoch` 完成一轮小批量训练；`train` 负责初始化并重复训练多轮，返回参数、训练历史、初始指标和最终指标。`train.py` 末尾调用辅助文件的 `main`，因此运行命令保持不变。

完整实验由 `experiment_utils.py` 中的 `run_experiment` 组织：先加载数据并验证实现，再调用核心 `train`，最后评估测试集并保存结果。绘图和验证的具体代码均在辅助文件中。

## 运行

在项目根目录运行（Python 3.10 或更新版本）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py
```

Windows PowerShell 的虚拟环境创建与激活命令为：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train.py
```

已在当前 WSL 环境建立 `.venv`，直接重跑可使用：

```bash
.venv/bin/python src/train.py
```

终端只显示每轮训练损失、训练准确率及最终训练/测试结果。验证结果和具体样本计算保存到文件。运行失败时仍会显示错误信息。

默认设置位于 `src/train.py` 开头：隐藏层宽度 128、64，学习率 0.1，batch_size 128，epochs 30，seed 42。命令行可显式指定：

```bash
python src/train.py --hidden1 128 --hidden2 64 --learning-rate 0.1 --batch-size 128 --epochs 30 --seed 42
```

`--data` 可指定本地 npz，`--output-dir` 可指定输出位置。默认读取本项目内已有 `mnist.npz`，仅在找不到本地文件时从 `https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz` 下载。下载失败会报错，不会替换数据集或生成虚假结果。默认重复运行会覆盖 `outputs` 中同名结果；保留多次实验请指定不同的输出目录。

## 1. 数据与 one-hot

对应 `src/load_data.py` 的 `load_data` 和 `one_hot`。

- 保留官方 60,000 张训练图与 10,000 张测试图的划分，标签覆盖全部 0～9。
- 原始图片为 `(n, 28, 28)` 的 uint8；转换为 float32 后展平并除以 255，得到 `(n, 784)`，像素范围为 `[0, 1]`。
- 整数标签 `y` 与 one-hot 标签 `T` 独立保留，不使用 -1/+1 标签。
- `one_hot(3) = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`。
- `X_train` 为 `(60000, 784)`，`T_train` 为 `(60000, 10)`；测试集形状对应为 `(10000, 784)` 和 `(10000, 10)`。
- 各类别数量保存到 `outputs/metrics.json`。

每轮用 `rng.permutation` 生成同一组索引，同时选取图片与 one-hot 标签。原整数标签用于评估，不在训练时单独打乱，因此仍与原图片顺序一致。训练只使用训练集，测试集不参与更新、轮数选择或超参数选择。

## 2. 全连接层

对应 `initialize` 和 `forward`。偏置单独保存，不给输入追加常数 1。

| 参数 | 形状 |
| --- | --- |
| W1、b1 | `(128, 784)`、`(128,)` |
| W2、b2 | `(64, 128)`、`(64,)` |
| W3、b3 | `(10, 64)`、`(10,)` |

单个样本是列向量时，`z = W @ a + b`。转置后得到 `z.T = a.T @ W.T + b.T`；将多个样本的转置堆成行，代码就是 `Z = A @ W.T + b`。权重每行仍对应一个输出神经元，因此使用 `W.T` 才能让输入特征与该神经元的权重做内积。NumPy 会沿样本维度广播偏置。

```python
Z1 = X @ W1.T + b1    # (m, 128)
A1 = sigmoid(Z1)
Z2 = A1 @ W2.T + b2   # (m, 64)
A2 = sigmoid(Z2)
Z3 = A2 @ W3.T + b3   # (m, 10)
```

Xavier 初始化让每个权重服从 `N(0, 2/(n_in+n_out))`。NumPy 的 `rng.normal` 接收标准差，所以传入 `sqrt(2/(n_in+n_out))`，偏置初始化为 0。默认网络共 109,386 个可训练参数。

## 3. sigmoid

对应 `sigmoid`。

```text
sigmoid(z) = 1 / (1 + exp(-z))
sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
```

数值实现对正负元素分开计算：`z >= 0` 使用原式，`z < 0` 使用等价的 `exp(z)/(1+exp(z))`。两条分支内的指数输入均不大于 0，避免指数溢出。反向传播缓存激活值 A，导数直接写成 `A * (1-A)`。

## 4. softmax

对应 `softmax_and_log_probs`，对每一行分别归一化：

```python
S = Z3 - Z3.max(axis=1, keepdims=True)
exp_S = np.exp(S)
P = exp_S / exp_S.sum(axis=1, keepdims=True)
log_P = S - np.log(exp_S.sum(axis=1, keepdims=True))
```

每行同时减同一常数 k 时，分子和分母都会乘以 `exp(-k)`，这个因子相消，因此概率不变。取行最大值作为 k，可以避免正指数溢出。P 的第 c 列表示数字 c 的概率，各概率非负，每行之和在浮点误差内为 1。

## 5. 交叉熵

对应 `cross_entropy`：

```text
单样本：ℓ = -Σc t[c] ln(p[c]) = -ln(p[y])
批次：L = -(1/m) Σi Σc T[i,c] ln(P[i,c])
```

one-hot 只有真实类别处为 1，因此求和会取出模型分给真实类别的概率。真实类别概率越小，负自然对数越大。比如该概率为 0.9 时损失约 0.105，为 0.1 时损失约 2.303。

实现中用 `-mean(sum(T * log_P, axis=1))`。稳定的 log-softmax 直接从分数计算对数概率，避免概率下溢到 0 后再取对数。这是单标签十分类的多分类交叉熵。

## 6. 反向传播

对应 `backward`，其中 `⊙` 表示逐元素乘法。对于单样本，softmax 与交叉熵合并求导可得：

```text
δ3 = p - t
δ2 = (W3.T @ δ3) ⊙ a2 ⊙ (1-a2)
δ1 = (W2.T @ δ2) ⊙ a1 ⊙ (1-a1)
∂ℓ/∂W_l = δ_l @ a_(l-1).T
∂ℓ/∂b_l = δ_l
```

链式法则先把输出层误差通过下一层的权重传回当前层，再乘当前 sigmoid 的局部导数。批次按行存储时，误差右乘 W，得到对应隐藏层各单元的误差：

```python
D3 = (P - T) / m                      # (m, 10)
dW3 = D3.T @ A2                       # (10, 64)
db3 = D3.sum(axis=0)                  # (10,)
D2 = (D3 @ W3) * A2 * (1-A2)          # (m, 64)
dW2 = D2.T @ A1                       # (64, 128)
db2 = D2.sum(axis=0)                  # (64,)
D1 = (D2 @ W2) * A1 * (1-A1)          # (m, 128)
dW1 = D1.T @ X                        # (128, 784)
db1 = D1.sum(axis=0)                  # (128,)
```

D3 已除以实际批次大小 m，后续梯度不再除。`P-T` 已含 softmax 的求导效果，不再额外乘 softmax 导数。所有梯度在参数更新前全部计算完毕。

## 7. 参数更新

对应 `update_parameters`、`train_epoch` 和 `train`：

```text
W_l ← W_l - learning_rate * dW_l
b_l ← b_l - learning_rate * db_l
```

每轮同步打乱训练样本，逐个小批次执行前向、损失、反向、更新。60,000 除以 128，最后一批有 96 张，照常参与计算，D3 用其实际样本数 96 归一化。没有 Adam、动量、正则化、Dropout、批归一化或学习率调度。

每轮结束后，`evaluate` 使用最新参数重新计算整个训练集损失与准确率。分批评估时累加 `批次平均损失 × 批次样本数`，最终除以总样本数；正确数量也逐批累加。CSV 的 epoch 0 记录随机初始化状态，epoch 1～30 为各轮结束后的状态。

正式训练前，`src/experiment_utils.py` 中的 `validate_implementation` 执行：

1. 在 7 张训练图片上验证前向形状、概率非负、概率和为 1、梯度形状及参数更新。
2. 使用 float64 和中心差分 `(L(θ+ε)-L(θ-ε))/(2ε)`，对各层权重和偏置抽样检查数值梯度，`ε=1e-5`；float64 仅用于检查，正式训练使用 float32。
3. 对 ±10000 等极端输入验证 sigmoid、softmax、交叉熵的稳定性。
4. 对 259 张训练图片做独立的短暂训练，确认损失总体下降，并覆盖最后不足 128 张的小批次。将不同评估批大小的结果互相核对，验证损失加权。

检查结束后丢弃该小模型，使用固定种子重新初始化正式模型。损失、梯度、参数在训练过程中持续检查有限性；任何检查失败都会终止并记录异常。

## 8. 分类预测

对应 `predict`：

```python
predicted_digit = np.argmax(P, axis=1)
accuracy = correct_count / sample_count
```

选择十个类别中概率最大的数字，不使用二分类的 0.5 阈值。全部训练完成后计算最终测试指标，并保存测试预测、混淆矩阵和图片。

测试图片展示按照真实数字 0～9 各选其第一次出现的图像，标注真实数字、预测数字及预测类别概率；如果存在错误预测，额外展示按原始顺序遇到的前十个错误样本。图片使用英文标签以兼容无中文字体的环境。

`src/experiment_utils.py` 中的 `explain_sample` 固定分析测试集第 0 张，保存真实标签、one-hot、各层形状、十个 logits、十个概率、概率和、真实类别概率、交叉熵与预测，并实际核对 `cross_entropy = -ln(p[y])`。同时输出 `p-t`：真实类别位置的 one-hot 是 1，所以该位置减去 1，其他位置减去 0。

## 结果文件

所有实验产物保存在 `outputs/`：

| 文件 | 内容 |
| --- | --- |
| `metrics.json` | 配置、数据哈希、环境版本、类别计数、最终训练/测试损失与准确率、正确数及用时 |
| `environment_requirements.txt` | 此次运行环境的完整依赖版本（pip freeze） |
| `validation.json` | 数值梯度误差、形状/稳定性检查、小数据训练前后结果 |
| `training_history.csv` | 每轮全训练集损失、准确率、轮耗时，含初始化 epoch 0 |
| `run.log` | 精简运行日志：每轮损失、训练准确率和最终训练/测试结果 |
| `model_parameters.npz` | W1、b1、W2、b2、W3、b3，可用 `np.load(..., allow_pickle=False)` 加载 |
| `training_loss.png` | 训练损失曲线 |
| `training_accuracy.png` | 训练准确率曲线 |
| `test_samples.png` | 真实标签覆盖 0～9 的十张测试图片 |
| `misclassified_samples.png` | 实际存在时输出前十个错误样本 |
| `test_predictions.npz` | 测试整数标签、预测数字、全部十类概率 |
| `confusion_matrix.csv` | 行为真实类别 0～9，列为预测类别 0～9 |
| `sample_calculation.json`、`sample_calculation.txt` | 测试集第 0 张的具体计算说明 |

加载模型并预测的例子（在根目录运行）：

```python
import sys
import numpy as np
sys.path.insert(0, "src")
from load_data import load_data
from train import predict

_, _, _, _, X_test, y_test, _ = load_data()
with np.load("outputs/model_parameters.npz", allow_pickle=False) as archive:
    params = {name: archive[name].copy() for name in archive.files}
digits, probabilities = predict(X_test[:10], params)
print("真实标签:", y_test[:10])
print("预测数字:", digits)
print("十类概率:", probabilities)
```

固定种子保证同一软件与计算环境下可以重复实验；不同 NumPy/BLAS 或硬件的浮点累积顺序可能造成末位差异。此次实际运行的依赖版本保存在 `outputs/metrics.json`。
