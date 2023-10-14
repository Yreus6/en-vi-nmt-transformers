import torch
from torch.utils.data import Dataset

from text_tokenizers.BPE.BPE_tokenizer import BPETokenizer

N = 30000


class NMTDataset(Dataset):
    def __init__(self, mode, src_data_path, tgt_data_path, src_vocab_path, tgt_vocab_path):
        self.src_tokenizer = BPETokenizer.get_tokenizer(src_vocab_path)
        self.tgt_tokenizer = BPETokenizer.get_tokenizer(tgt_vocab_path)
        self.mode = mode

        with open(src_data_path, 'r') as f_src, open(tgt_data_path, 'r') as f_tgt:
            src_lines = f_src.readlines()
            tgt_lines = f_tgt.readlines()
            assert len(src_lines) == len(tgt_lines)
            self.src_tgt_pairs = list(zip(src_lines[:N], tgt_lines[:N]))

    def __len__(self):
        return len(self.src_tgt_pairs)

    def __getitem__(self, idx):
        src, tgt = self.src_tgt_pairs[idx]

        if self.mode == 'test':
            return src, tgt

        src = self.src_tokenizer.encode(src.rstrip('\n'))
        tgt = self.tgt_tokenizer.encode(tgt.rstrip('\n'))

        return torch.tensor(src.ids), torch.tensor(tgt.ids)
