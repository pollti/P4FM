import math
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
        self._legend()
        for file_type in self._file_types:
            self._fig.savefig('%s/%s.%s' % (self._out_dir, self._file_name, file_type), dpi=200)
        plt.close(self._fig)

    def _legend(self, force_place_legend_inside: bool = False):
        axes = self._fig.get_axes()
        if force_place_legend_inside or not self._place_legend_outside:
            for ax in axes:
                ax.legend(loc='upper left')
        else:
            handles = []
            labels = []
            for ax in axes:
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    if label not in labels:
                        handles.append(handle)
                        labels.append(label)
            handles = handles
            labels = labels
            if self._nrows == 1:
                pos = 0.9
                pos -= 0.05 * (int(math.ceil(len(handles) / 3)) - 1)
            elif self._nrows == 2:
                pos = 0.925
                pos -= 0.025 * (int(math.ceil(len(handles) / 3)) - 1)
            elif self._nrows == 3:
                pos = 0.95
                pos -= 0.025 * (int(math.ceil(len(handles) / 3)) - 1)
            else:
                print(f'Unsupported number of rows: {self._nrows} Legend will be placed in every axes!')
                return self._legend(force_place_legend_inside=True)
            self._fig.subplots_adjust(top=pos)
            self._fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, pos), ncol=3)


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
