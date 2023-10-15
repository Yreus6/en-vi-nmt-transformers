import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataset.nmt_dataset import NMTDataset
from models.transformers.utils import PAD_IDX
from text_tokenizers.BPE.BPE_tokenizer import BPETokenizer


def loading_data(args):
    src_tokenizer = BPETokenizer.get_tokenizer(args.src_vocab_path)
    tgt_tokenizer = BPETokenizer.get_tokenizer(args.tgt_vocab_path)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample.rstrip('\n'))
            tgt_batch.append(tgt_sample.rstrip('\n'))

        src_batch = [torch.tensor(x.ids) for x in src_tokenizer.encode_batch(src_batch)]
        tgt_batch = [torch.tensor(x.ids) for x in tgt_tokenizer.encode_batch(tgt_batch)]
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

        return src_batch, tgt_batch

    train_set = NMTDataset(
        src_data_path=args.train_src_data_path,
        tgt_data_path=args.train_tgt_data_path
    )
    val_set = NMTDataset(
        src_data_path=args.val_src_data_path,
        tgt_data_path=args.val_tgt_data_path
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=False
    )

    return train_loader, val_loader
