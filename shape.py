import numpy as np
from pathlib import Path

feat_dir = Path("./imagenet_feature/imagenet256_features")
lab_dir  = Path("./imagenet_feature/imagenet256_labels")

key = "0_0"
x = np.load(feat_dir / (key + ".npy"), allow_pickle=False)          # or feat_dir/(key+".npy")
y = np.load(lab_dir  / (key + ".npy"), allow_pickle=False)

print("x dtype/shape:", x.dtype, x.shape)
print("y dtype/shape:", y.dtype, y.shape)
print("first labels:", y[:10])
