"""神经网络核心：初始化、前向传播、反向传播、参数更新与训练。"""
import os

# 小矩阵乘法使用单线程，用户可在运行前自行覆盖。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time

import numpy as np

from experiment_utils import ensure_finite, log_epoch

# 默认实验设置；命令行参数也可以覆盖。
HIDDEN1 = 128
HIDDEN2 = 64
LEARNING_RATE = 0.1
BATCH_SIZE = 128
EPOCHS = 30
RANDOM_SEED = 42
EVAL_BATCH_SIZE = 1024


def initialize(rng, hidden1=HIDDEN1, hidden2=HIDDEN2, dtype=np.float32):
    """Xavier 正态初始化；normal 的第二个参数是标准差。"""
    params = {}
    widths = [784, hidden1, hidden2, 10]
    for layer, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:]), start=1):
        std = np.sqrt(2.0 / (n_in + n_out))
        params[f"W{layer}"] = rng.normal(0, std, (n_out, n_in)).astype(dtype)
        params[f"b{layer}"] = np.zeros(n_out, dtype=dtype)
    return params


def sigmoid(z):
    """按正负分支计算，避免 exp 对极大正数溢出。"""
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    return result


def softmax_and_log_probs(z):
    """每行减最大值；直接生成稳定的 log_softmax。"""
    shifted = z - z.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    denominator = exp_shifted.sum(axis=1, keepdims=True)
    probabilities = exp_shifted / denominator
    log_probabilities = shifted - np.log(denominator)
    return probabilities, log_probabilities


def cross_entropy(log_probabilities, targets):
    """L = -mean(sum(T * log_P))，使用自然对数。"""
    per_sample = -np.sum(targets * log_probabilities, axis=1)
    # 用 float64 累加，但网络矩阵运算保留 float32。
    loss = float(np.mean(per_sample, dtype=np.float64))
    ensure_finite("交叉熵", loss)
    return loss


def forward(X, params):
    """一行一个样本；W 的每一行是一个输出神经元的权重。"""
    Z1 = X @ params["W1"].T + params["b1"]  # (m, hidden1)
    A1 = sigmoid(Z1)                          # (m, hidden1)
    Z2 = A1 @ params["W2"].T + params["b2"] # (m, hidden2)
    A2 = sigmoid(Z2)                          # (m, hidden2)
    Z3 = A2 @ params["W3"].T + params["b3"] # (m, 10)
    P, log_P = softmax_and_log_probs(Z3)       # (m, 10)
    return {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2,
            "Z3": Z3, "P": P, "log_P": log_P}


def backward(cache, targets, params):
    """用更新前的同一组参数算完全部梯度；仅在 D3 中除一次 m。"""
    X, A1, A2, P = (cache[key] for key in ["X", "A1", "A2", "P"])
    m = X.shape[0]
    D3 = (P - targets) / m                 # (m, 10)
    dW3 = D3.T @ A2                       # (10, hidden2)
    db3 = D3.sum(axis=0)                  # (10,)
    D2 = (D3 @ params["W3"]) * A2 * (1 - A2)  # (m, hidden2)
    dW2 = D2.T @ A1                       # (hidden2, hidden1)
    db2 = D2.sum(axis=0)                  # (hidden2,)
    D1 = (D2 @ params["W2"]) * A1 * (1 - A1)  # (m, hidden1)
    dW1 = D1.T @ X                        # (hidden1, 784)
    db1 = D1.sum(axis=0)                  # (hidden1,)
    gradients = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                 "W3": dW3, "b3": db3}
    ensure_finite("梯度", *gradients.values())
    return gradients


def update_parameters(params, gradients, learning_rate):
    """普通小批量梯度下降：参数 ← 参数 - 学习率 × 梯度。"""
    for name in params:
        params[name] -= learning_rate * gradients[name]
    ensure_finite("参数", *params.values())


def predict(X, params):
    """十分类使用 argmax，返回整数数字及十类概率。"""
    probabilities = forward(X, params)["P"]
    return probabilities.argmax(axis=1), probabilities


def evaluate(X, labels, targets, params, batch_size=EVAL_BATCH_SIZE):
    """分批评估；按样本数加权损失，避免最后一批权重错误。"""
    total_loss, correct = 0.0, 0
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        cache = forward(X[start:stop], params)
        total_loss += cross_entropy(cache["log_P"], targets[start:stop]) * (stop - start)
        correct += int(np.count_nonzero(cache["P"].argmax(axis=1) == labels[start:stop]))
    return {"loss": total_loss / len(X), "accuracy": correct / len(X),
            "correct": correct, "total": len(X)}


def train_epoch(X, targets, params, rng, batch_size, learning_rate):
    """同一索引同步打乱图片与标签，最后不足一批的样本仍参与。"""
    order = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        indices = order[start:start + batch_size]
        cache = forward(X[indices], params)
        cross_entropy(cache["log_P"], targets[indices])
        gradients = backward(cache, targets[indices], params)
        update_parameters(params, gradients, learning_rate)


def train(X, labels, targets, hidden1=HIDDEN1, hidden2=HIDDEN2,
          learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
          epochs=EPOCHS, seed=RANDOM_SEED):
    """从初始化开始训练；返回模型参数、训练记录、初始和最终训练指标。"""
    rng = np.random.default_rng(seed)
    params = initialize(rng, hidden1, hidden2)
    initial = evaluate(X, labels, targets, params)
    history = [{"epoch": 0, "train_loss": initial["loss"],
                "train_accuracy": initial["accuracy"], "seconds": 0.0}]

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_epoch(X, targets, params, rng, batch_size, learning_rate)
        metrics = evaluate(X, labels, targets, params)
        history.append({"epoch": epoch, "train_loss": metrics["loss"],
                        "train_accuracy": metrics["accuracy"],
                        "seconds": time.perf_counter() - epoch_start})
        log_epoch(epoch, epochs, metrics)

    return params, history, initial, metrics


if __name__ == "__main__":
    # 参数解析、验证、绘图和保存交给辅助文件，阅读本文件时可先关注上面的计算。
    from experiment_utils import main
    main()
