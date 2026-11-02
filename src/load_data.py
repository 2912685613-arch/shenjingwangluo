"""读取官方 MNIST；保留官方训练/测试划分，仅用 NumPy 处理数据。"""
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def one_hot(labels, num_classes=10):
    """整数标签转 one-hot；原整数标签仍独立保留。"""
    return np.eye(num_classes, dtype=np.float32)[labels]


def load_data(data_path=None):
    """先找本地 mnist.npz；找不到才下载官方 MNIST 的 npz 封装。"""
    if data_path is not None:
        path = Path(data_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"指定的数据文件不存在：{path}")
    else:
        common_paths = [PROJECT_ROOT / "data/mnist_data/mnist.npz",
                        PROJECT_ROOT / "data/mnist.npz", PROJECT_ROOT / "mnist.npz"]
        candidates = [candidate for candidate in common_paths if candidate.is_file()]
        if not candidates:
            candidates = sorted(PROJECT_ROOT.rglob("mnist.npz"))
        path = candidates[0] if candidates else common_paths[0]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".download")
            try:
                urlretrieve(MNIST_URL, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    with np.load(path, allow_pickle=False) as archive:
        images_train = archive["x_train"].copy()
        images_test = archive["x_test"].copy()
        y_train = archive["y_train"].astype(np.int64)
        y_test = archive["y_test"].astype(np.int64)
    for images, labels, count in [(images_train, y_train, 60000), (images_test, y_test, 10000)]:
        if images.shape != (count, 28, 28) or labels.shape != (count,):
            raise ValueError("必须使用官方 MNIST 的 60000/10000 划分、28×28 图片。")
        if images.dtype != np.uint8 or not np.array_equal(np.unique(labels), np.arange(10)):
            raise ValueError("MNIST 图片必须为 uint8，标签必须覆盖 0～9。")
    X_train = images_train.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = images_test.reshape(-1, 784).astype(np.float32) / 255.0
    return path, X_train, y_train, one_hot(y_train), X_test, y_test, one_hot(y_test)
