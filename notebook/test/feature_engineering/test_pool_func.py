#%%
from functools import partial

import audio_classifier.common.feature_engineering.skm.pool as skm_pool
import numpy as np

#%%
arr = np.array([[2, 7, 9], [5, 8, 0], [2, 6, 6], [2, 8, 8], [0, 8, 8]],
               dtype=np.float64)
pool_func = skm_pool.PoolFunc(
    (partial(np.mean, axis=0,
             keepdims=True), partial(np.std, axis=0, keepdims=True)))
res = skm_pool.apply_pool_func(arr, pool_func, pool_size=3, stride_size=1)
print(res)

# %%
arr = np.array([[2, 7, 9], [5, 8, 0], [2, 6, 6], [2, 8, 8], [0, 8, 8]],
               dtype=np.float64)
pool_func = skm_pool.PoolFunc(
    (partial(np.mean, axis=0,
             keepdims=True), partial(np.std, axis=0, keepdims=True)))
res = skm_pool.apply_pool_func(arr, pool_func)
print(res)
