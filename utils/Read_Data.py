
from pathlib import Path
from typing import Union, Iterable, Iterator, Generator, Dict
import dill

def read_line(*, file: Union[Path, str]) -> Generator:
    """Reads a data file line by line """
    with open(file, mode='r') as f:
        for line in f:
            yield line
            
def load_pkl(*, file: Union[Path, str]):
    """Loads a pickled object """
    with open(file, "rb") as f:
        data = dill.load(f)
        return data