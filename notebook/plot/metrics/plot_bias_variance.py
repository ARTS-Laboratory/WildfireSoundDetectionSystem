#%%
import pickle
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

#%%
plt.style.use(["science", "ieee"])

#%%
METRICS_DIR_PATH: str = "../../../metrics/binary_fire/spec_00_reshape_00_skm_00/bias_variance/svc_04"
DATA_PER_FOLD: int = 588


#%%
def load_accs(accs_path: str):
    with open(accs_path, mode="rb") as acc_file:
        accs: Tuple[List[float], List[float],
                    List[float]] = pickle.load(acc_file)
        return accs


#%%
filenames: List[str] = os.listdir(METRICS_DIR_PATH)
filenames = list(
    filter(lambda fname: os.path.splitext(fname)[1] == ".pkl", filenames))
for filename in filenames:
    accs_path: str = os.path.join(METRICS_DIR_PATH, filename)
    train_accs, val_accs, test_accs = load_accs(accs_path)
    plt.plot(np.linspace(588, 588 * len(train_accs), len(train_accs)),
             train_accs,
             label="Train")
    plt.plot(np.linspace(588, 588 * len(val_accs), len(val_accs)),
             val_accs,
             label="Validation")
    plt.plot(np.linspace(588, 588 * len(test_accs), len(test_accs)),
             test_accs,
             label="Test")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Training Samples")
    plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
    cf = plt.gcf()
    cf.set_size_inches(cf.get_size_inches()[0] * 2, cf.get_size_inches()[1])
    plt.tight_layout()
    plt.show()
    fig_filename: str = os.path.splitext(filename)[0] + ".png"
    fig_path: str = os.path.join(METRICS_DIR_PATH, fig_filename)
    plt.savefig(fig_path, dpi=300)

# %%
