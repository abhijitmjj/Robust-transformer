import os


import jax
from jax import vmap, grad, jacfwd, jacrev, jit, lax
import jax.numpy as jnp
from jax import random, device_put

from pathlib import Path
from typing import Union, Iterable, Iterator, Generator, Dict
import collections
import toolz
import dill
import numpy as np

import pandas as pd

def split_shuffle(*, key: jax.numpy.lax_numpy.ndarray,
                  raw_data: pd.DataFrame, n: int) -> Dict[str, pd.DataFrame]:
    """Yields a dictionary of n fold splits"""

    permutation = random.permutation(key, jnp.arange(len(raw_data)))
    mark = len(raw_data)//n
    for i in range(n):
        mask = np.zeros(len(raw_data), dtype=bool)
        mask[np.arange(i*mark, (i+1)*mark)] = True
        yield {"Test": raw_data.iloc[permutation[mask], :],
               "Train": raw_data.iloc[permutation[~mask], :]}
        
def inner_split(*, key: jax.numpy.lax_numpy.ndarray,
                outer_fold_data: Dict[str, pd.DataFrame], n: int) -> Dict[str, pd.DataFrame]:
    """Yields inner split for parameter tuning"""
    train_data: pd.DataFrame = outer_fold_data["Train"]
    permutation = random.permutation(key, jnp.arange(len(train_data)))
    mark = len(train_data)//n
    for j in range(n):
        mask = np.zeros(len(train_data), dtype=bool)
        mask[np.arange(j*mark, (j+1)*mark)] = True
        yield {"Train": train_data.iloc[permutation[~mask], :],
               "Val": train_data.iloc[permutation[mask], :]}
        
def outer_loop( *, key: jax.numpy.lax_numpy.ndarray,
                  raw_data: pd.DataFrame, n: int) -> Dict[str, pd.DataFrame]:
    """Yields a dictionary of n fold splits"""

    permutation = random.permutation(key, jnp.arange(len(raw_data)))
    mark = len(raw_data)//n
    for i in range(n):
        mask = np.zeros(len(raw_data), dtype=bool)
        mask[np.arange(i*mark, (i+1)*mark)] = True
        yield {"key": key, "outer_fold_data":{"Test": raw_data.iloc[permutation[mask], :],
               "Train": raw_data.iloc[permutation[~mask], :]}, "n": n}

        
def inner_loop( *, key: jax.numpy.lax_numpy.ndarray,
                outer_fold_data: Dict[str, pd.DataFrame], n: int) -> Dict[str, pd.DataFrame]:
    """Yields inner split for parameter tuning"""
    train_data: pd.DataFrame = outer_fold_data["Train"]
    permutation = random.permutation(key, jnp.arange(len(train_data)))
    mark = len(train_data)//n
    for j in range(n):
        mask = np.zeros(len(train_data), dtype=bool)
        mask[np.arange(j*mark, (j+1)*mark)] = True
        yield {"Train": train_data.iloc[permutation[~mask], :],
               "Val": train_data.iloc[permutation[mask], :]}
        
