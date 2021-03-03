# With the help of https://github.com/fdamken/bachelors-thesis_code/blob/cd69af4d1e385e6b91b3bc12bd97c60671857e30/investigation/plot_util.py
# Permission granted by author.

import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt


class SubplotsAndSave:
    _out_dir: str
    _file_name: str
    file_types: List[str]

    def __init__(self, out_dir: str, file_name: str, /, nrows: int = 1, ncols: int = 1, place_legend_outside: bool = False, file_types: Optional[List[str]] = None, **kwargs):
        if file_types is None:
            file_types = ['png', 'pdf']

        self._out_dir = out_dir
        self._file_name = file_name
        self._file_types = file_types
        self._nrows = nrows
        self._ncols = ncols
        self._place_legend_outside = place_legend_outside
        self._kwargs = kwargs

        os.makedirs(out_dir, exist_ok=True)

    def __enter__(self):
        self._fig, self._axs = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=figsize(self._nrows, self._ncols, self._place_legend_outside), **self._kwargs)
        return self._fig, self._axs

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fig.tight_layout()
        for file_type in self._file_types:
            self._fig.savefig('%s/%s.%s' % (self._out_dir, self._file_name, file_type), dpi=200)
        plt.close(self._fig)


def figsize(nrows: int, ncols: int, place_legend_outside: bool = False) -> Tuple[int, int]:
    if place_legend_outside:
        if nrows == 1:
            pad = 1
        elif nrows == 2:
            pad = 1.5
        elif nrows == 3:
            pad = 2
        else:
            print(f'Unsupported number of rows: {nrows} Legend will be placed in every axes when using SubplotsAndSave!')
            pad = 0
    else:
        pad = 0
    return 2 + 5 * ncols, 1 + 4 * nrows + pad


def even(x: float) -> int:
    n = int(x)
    n += n % 2
    return n
