from torch.utils.data import Dataset


class NMTDataset(Dataset):
    def __init__(self, src_data_path, tgt_data_path):
        with open(src_data_path, encoding='utf-8') as f_src, open(tgt_data_path, encoding='utf-8') as f_tgt:
            src_lines = f_src.readlines()
            tgt_lines = f_tgt.readlines()
            assert len(src_lines) == len(tgt_lines)
            self.src_tgt_pairs = list(zip(src_lines, tgt_lines))

    def __len__(self):
        return len(self.src_tgt_pairs)

    def __getitem__(self, idx):
        return self.src_tgt_pairs[idx]
