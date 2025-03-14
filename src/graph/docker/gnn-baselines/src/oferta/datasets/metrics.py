from pathlib import Path
from collections import Counter

import torch
from torch_geometric.data import Data


class FeatureMean(object):
    def __init__(self, key):
        super().__init__()
        self.key = key
        self.curr_avg = None
        self.total = 0

    def update(self, data: Data):
        """
        Calculate mean over a stream of numbers/vectors. Avoids storing a potentially very large sum explicitly.

        Update rule for new value x:
        avg := (avg * (n-1) + x) / n
            := (n-1)/n * avg + x/n
        """
        val = data[self.key]
        prev_num = self.total
        curr_num = 1
        if len(val.shape) == 2:
            curr_num = val.shape[0]
            val = val.sum(dim=0)

        new_total = prev_num + curr_num
        if self.curr_avg is None:
            self.curr_avg = val / new_total
        else:
            self.curr_avg = (prev_num / new_total) * self.curr_avg + val / new_total
        self.total = new_total

    def compute(self):
        return self.curr_avg


class FeatureStd(object):
    def __init__(self, key, mean):
        super().__init__()
        self.key = key
        self.mean = mean
        self.curr_d2 = None
        self.total = 0

    def update(self, data: Data):
        val = data[self.key]
        prev_num = self.total
        curr_num = 1

        d2 = torch.pow(torch.abs(val - self.mean), 2)
        if len(d2.shape) == 2:
            curr_num = d2.shape[0]
            d2 = d2.sum(dim=0)

        new_total = prev_num + curr_num
        if self.curr_d2 is None:
            self.curr_d2 = d2 / new_total
        else:
            self.curr_d2 = (prev_num / new_total) * self.curr_d2 + d2 / new_total
        self.total = new_total

    def compute(self):
        return torch.sqrt(self.curr_d2)


class LabelDist(object):
    def __init__(self):
        super().__init__()
        self.label_count = Counter()
        self.total = 0

    def update(self, in_file: Path):
        label = int(in_file.stem.split("_")[0])
        self.label_count[label] += 1
        self.total += 1

    def compute(self):
        counts = []
        for i in range(max(self.label_count.keys()) + 1):
            counts.append(self.label_count[i])
        counts = torch.tensor(counts, dtype=torch.float)
        return counts / self.total
