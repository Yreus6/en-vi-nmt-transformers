from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from datasets.nmt_dataset import NMTDataset
from models.transformers.utils import PAD_IDX


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return src_batch, tgt_batch


def loading_data(args):
    train_set = NMTDataset(
        mode='train',
        src_data_path=args.train_src_data_path,
        tgt_data_path=args.train_tgt_data_path,
        src_vocab_path=args.src_vocab_path,
        tgt_vocab_path=args.tgt_vocab_path
    )
    val_set = NMTDataset(
        mode='val',
        src_data_path=args.val_src_data_path,
        tgt_data_path=args.val_tgt_data_path,
        src_vocab_path=args.src_vocab_path,
        tgt_vocab_path=args.tgt_vocab_path
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
