"""实验辅助：正确性验证、绘图、日志、命令行与结果文件保存。"""
import argparse
import csv
import hashlib
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np

from load_data import PROJECT_ROOT, load_data


def ensure_finite(name, *arrays):
    """损失、梯度、参数均不得出现 NaN 或无穷大。"""
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError(f"{name} 中出现 NaN 或无穷大")


def validate_implementation(X, labels, targets, seed, hidden1, hidden2):
    """训练前检验公式、数值稳定性、梯度以及小数据能否学习。"""
    # 在调用时导入核心函数，避免两个文件在导入阶段互相依赖。
    from train import (initialize, forward, backward, cross_entropy, sigmoid,
                       softmax_and_log_probs, update_parameters, evaluate, train_epoch)

    rng = np.random.default_rng(seed)
    params = initialize(rng, hidden1, hidden2, dtype=np.float64)
    xb, tb = X[:7].astype(np.float64), targets[:7].astype(np.float64)
    cache = forward(xb, params)
    expected_shapes = {"X": (7, 784), "Z1": (7, hidden1), "A1": (7, hidden1),
                       "Z2": (7, hidden2), "A2": (7, hidden2), "Z3": (7, 10), "P": (7, 10)}
    for key, shape in expected_shapes.items():
        assert cache[key].shape == shape, (key, cache[key].shape)
    np.testing.assert_allclose(cache["P"].sum(axis=1), 1, atol=1e-12)
    assert np.all(cache["P"] >= 0)
    gradients = backward(cache, tb, params)
    for name in params:
        assert gradients[name].shape == params[name].shape
    # 中心差分逐个扰动参数，独立验证手推反向传播；不用自动求导。
    epsilon, maximum_error, checked = 1e-5, 0.0, 0
    for name, value in params.items():
        indices = rng.choice(value.size, size=min(12, value.size), replace=False)
        for index in indices:
            old = value.flat[index]
            value.flat[index] = old + epsilon
            plus = cross_entropy(forward(xb, params)["log_P"], tb)
            value.flat[index] = old - epsilon
            minus = cross_entropy(forward(xb, params)["log_P"], tb)
            value.flat[index] = old
            numerical = (plus - minus) / (2 * epsilon)
            analytical = gradients[name].flat[index]
            error = abs(numerical - analytical)
            maximum_error = max(maximum_error, float(error))
            np.testing.assert_allclose(analytical, numerical, rtol=2e-4, atol=2e-8)
            checked += 1
    before = {name: value.copy() for name, value in params.items()}
    update_parameters(params, gradients, 0.1)
    for name in params:
        np.testing.assert_allclose(params[name], before[name] - 0.1 * gradients[name])
        assert np.any(params[name] != before[name])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values = sigmoid(np.array([-10000., -1000., 0., 1000., 10000.]))
        np.testing.assert_allclose(values, [0, 0, .5, 1, 1])
        p, log_p = softmax_and_log_probs(np.array([[10000., 0., -10000.]]))
        ensure_finite("极端分数", p, log_p)
        assert cross_entropy(log_p, np.array([[0., 0., 1.]])) == 20000.
    # 用 259 张训练图像，批大小 128，特意检验最后不足一批的情况。
    small_X, small_y, small_T = X[:259], labels[:259], targets[:259]
    small_rng = np.random.default_rng(seed)
    small_params = initialize(small_rng, hidden1, hidden2)
    initial = evaluate(small_X, small_y, small_T, small_params, batch_size=128)
    # 独立比较分批与整批评估，检验按样本数加权。
    whole = evaluate(small_X, small_y, small_T, small_params, batch_size=259)
    np.testing.assert_allclose(initial["loss"], whole["loss"], rtol=1e-6)
    assert initial["correct"] == whole["correct"]
    for _ in range(60):
        train_epoch(small_X, small_T, small_params, small_rng, 128, 0.1)
    final = evaluate(small_X, small_y, small_T, small_params, batch_size=128)
    assert final["loss"] < initial["loss"] * .95, (initial, final)
    return {"gradient_coordinates_checked": checked, "gradient_max_absolute_error": maximum_error,
            "forward_shapes": {key: list(shape) for key, shape in expected_shapes.items()},
            "probability_sum_check": "passed", "gradient_shape_and_update_check": "passed",
            "numerical_stability_check": "passed", "weighted_evaluation_check": "passed",
            "small_training": {"samples": 259, "epochs": 60, "batch_size": 128,
                               "learning_rate": .1, "initial": initial, "final": final}}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def log_epoch(epoch, epochs, metrics):
    """每轮只显示训练损失和准确率。"""
    logging.getLogger("mnist").info(
        "第 %02d/%d 轮 | 损失 %.4f | 训练准确率 %.2f%%",
        epoch, epochs, metrics["loss"], metrics["accuracy"] * 100)


def save_training_history(output_dir, history):
    """将初始化和每轮的训练指标写入 CSV。"""
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_results(output_dir, history, X_test, y_test, predictions, probabilities):
    """绘图采用英文标签，避免运行环境缺少中文字体。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    for field, ylabel, filename in [("train_loss", "Training cross-entropy", "training_loss.png"),
                                    ("train_accuracy", "Training accuracy", "training_accuracy.png")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, [row[field] for row in history], marker=".")
        ax.set(xlabel="Epoch (0 = initialization)", ylabel=ylabel)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    def image_grid(indices, filename):
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for ax, index in zip(axes.flat, indices):
            digit = int(predictions[index])
            ax.imshow(X_test[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"True {y_test[index]}, pred {digit}\np(pred)={probabilities[index, digit]:.4f}",
                         color="green" if digit == y_test[index] else "red", fontsize=10)
            ax.axis("off")
        for ax in list(axes.flat)[len(indices):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    # 按真实标签各取第一张，稳定地覆盖 0～9，不挑选表现最好的样本。
    samples = [int(np.flatnonzero(y_test == digit)[0]) for digit in range(10)]
    image_grid(samples, "test_samples.png")
    errors = np.flatnonzero(predictions != y_test)
    if len(errors):
        image_grid(errors[:10].tolist(), "misclassified_samples.png")
    return samples, errors


def explain_sample(X, labels, targets, params, index, output_dir):
    from train import forward, cross_entropy

    cache = forward(X[index:index + 1], params)
    p, t = cache["P"][0], targets[index]
    label = int(labels[index])
    ce = cross_entropy(cache["log_P"], targets[index:index + 1])
    negative_log_probability = float(-np.log(float(p[label])))
    np.testing.assert_allclose(ce, negative_log_probability, rtol=1e-6, atol=1e-7)
    result = {"test_index": index, "true_digit": label, "one_hot": t.tolist(),
              "shapes": {key: list(cache[key].shape) for key in ["X", "Z1", "A1", "Z2", "A2", "Z3", "P"]},
              "logits": cache["Z3"][0].tolist(), "probabilities": p.tolist(),
              "probability_sum": float(p.sum(dtype=np.float64)),
              "true_class_probability": float(p[label]), "cross_entropy": ce,
              "negative_log_true_class_probability": negative_log_probability,
              "cross_entropy_identity_verified": True, "predicted_digit": int(p.argmax()),
              "output_error_p_minus_t": (p - t).tolist(),
              "explanation": "one-hot 只有真实类别位置为 1，因此 p-t 在该位置减去 1；其他位置减去 0。"}
    write_json(output_dir / "sample_calculation.json", result)
    (output_dir / "sample_calculation.txt").write_text(
        "具体测试样本的前向传播与输出层误差（按类别 0～9 排列）\n"
        + json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run_experiment(args, output_dir):
    """组织数据、训练前验证、正式训练、最终测试与结果保存。"""
    import matplotlib
    from train import train, evaluate, predict

    log = logging.getLogger("mnist")
    start_time = time.perf_counter()
    path, X_train, y_train, T_train, X_test, y_test, T_test = load_data(args.data)
    validation = validate_implementation(X_train, y_train, T_train, args.seed, args.hidden1, args.hidden2)
    write_json(output_dir / "validation.json", validation)
    # 验证模型丢弃；核心 train 函数按固定种子重新初始化正式模型。
    params, history, initial, metrics = train(
        X_train, y_train, T_train, hidden1=args.hidden1, hidden2=args.hidden2,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        epochs=args.epochs, seed=args.seed)
    save_training_history(output_dir, history)
    np.savez_compressed(output_dir / "model_parameters.npz", **params)
    # 全部训练结束后才计算测试集指标，不用测试集选择参数或轮数。
    test_metrics = evaluate(X_test, y_test, T_test, params)
    predictions, probabilities = predict(X_test, params)
    np.savez_compressed(output_dir / "test_predictions.npz", labels=y_test, predictions=predictions,
                        probabilities=probabilities)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (y_test, predictions), 1)
    np.savetxt(output_dir / "confusion_matrix.csv", confusion, fmt="%d", delimiter=",",
               header="pred_0,pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9", comments="")
    samples, errors = plot_results(output_dir, history, X_test, y_test, predictions, probabilities)
    explain_sample(X_test, y_test, T_test, params, 0, output_dir)
    configuration = {"hidden1": args.hidden1, "hidden2": args.hidden2, "learning_rate": args.learning_rate,
                     "batch_size": args.batch_size, "epochs": args.epochs, "seed": args.seed}
    results = {"config": configuration, "dataset_path": str(path),
               "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "class_counts": {"train": np.bincount(y_train, minlength=10).tolist(),
                                "test": np.bincount(y_test, minlength=10).tolist()},
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "matplotlib": matplotlib.__version__, "platform": platform.platform()},
               "parameter_count": sum(value.size for value in params.values()),
               "initial_train": initial,
               "final_train": {"loss": history[-1]["train_loss"], "accuracy": history[-1]["train_accuracy"],
                               "correct": metrics["correct"], "total": len(X_train)},
               "test": test_metrics, "misclassified_count": len(errors),
               "plotted_test_indices": samples, "plotted_error_indices": errors[:10].tolist(),
               "elapsed_seconds": time.perf_counter() - start_time}
    write_json(output_dir / "metrics.json", results)
    log.info("最终训练准确率：%.4f%% (%d/%d)", results["final_train"]["accuracy"] * 100,
             results["final_train"]["correct"], len(X_train))
    log.info("最终测试准确率：%.4f%% (%d/%d)，交叉熵 %.6f", test_metrics["accuracy"] * 100,
             test_metrics["correct"], test_metrics["total"], test_metrics["loss"])


def main():
    """解析运行参数、配置日志，然后启动完整实验。"""
    from train import HIDDEN1, HIDDEN2, LEARNING_RATE, BATCH_SIZE, EPOCHS, RANDOM_SEED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--hidden1", type=int, default=HIDDEN1)
    parser.add_argument("--hidden2", type=int, default=HIDDEN2)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if min(args.hidden1, args.hidden2, args.batch_size, args.epochs) <= 0:
        parser.error("隐藏层宽度、批大小和轮数必须是正整数")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("学习率必须为有限正数")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8"), logging.StreamHandler()])
    try:
        run_experiment(args, output_dir)
    except Exception:
        logging.getLogger("mnist").exception("运行失败，未生成的指标不会被编造。")
        raise
