import os
import sys
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

SEED = 42

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

if __name__ == "__main__":
    path = sys.argv[1]

    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = list(kfold.split(train_df))[0]
    train_df, val_df = train_df.iloc[train_idx].reset_index(), train_df.iloc[val_idx].reset_index()
    train_df.to_csv(os.path.join(path, 'train.csv'))
    val_df.to_csv(os.path.join(path, 'val.csv'))