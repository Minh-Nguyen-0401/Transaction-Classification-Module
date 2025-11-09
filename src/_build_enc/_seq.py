import numpy as np
import torch
from torch.utils.data import Dataset


class SeqDS(Dataset):
    """Sequential Dataset for sender transaction history"""
    def __init__(self, grouped_df, maxlen=100):
        self.samples = []
        for _, sub in grouped_df:
            a = sub[["amount_log","hour_sin","hour_cos"]].to_numpy(dtype=np.float32)
            t = sub["tt_idx"].to_numpy(dtype=np.int64)
            c = sub["ch_idx"].to_numpy(dtype=np.int64)
            y_t = sub["tt_idx"].to_numpy(dtype=np.int64)
            y_a = sub["amt_bin"].to_numpy(dtype=np.int64)
            
            if len(a) < 2:
                continue
                
            for i in range(1, len(a)):
                s = max(0, i - maxlen)
                self.samples.append((a[s:i], t[s:i], c[s:i], y_t[i], y_a[i]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        return self.samples[i]


def collate_seq(batch):
    """Collate function for variable-length sequences"""
    lens = [len(b[0]) for b in batch]
    m = max(lens)
    xn = []
    xt = []
    xc = []
    
    for a, t, c, _, _ in batch:
        pad = m - len(a)
        xn.append(np.pad(a, ((0, pad), (0, 0))))
        xt.append(np.pad(t, (0, pad)))
        xc.append(np.pad(c, (0, pad)))
    
    xn = torch.tensor(np.stack(xn), dtype=torch.float32)
    xt = torch.tensor(np.stack(xt), dtype=torch.long)
    xc = torch.tensor(np.stack(xc), dtype=torch.long)
    yt = torch.tensor([b[3] for b in batch], dtype=torch.long)
    ya = torch.tensor([b[4] for b in batch], dtype=torch.long)
    
    return xn, xt, xc, yt, ya
