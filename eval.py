import sys
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.nmt_dataset import NMTDataset
from models.transformers.seq2seq_trans import seq2seq_trans
from models.transformers.utils import DEVICE, translate
from sacrebleu.metrics import BLEU


def parse_args():
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--test-src-data-path', default='./data',
                        help='training src data file')
    parser.add_argument('--test-tgt-data-path', default='./data',
                        help='training tgt data file')
    parser.add_argument('--src-vocab-path', default='./data',
                        help='src vocab data file')
    parser.add_argument('--tgt-vocab-path', default='./data',
                        help='tgt vocab data file')
    parser.add_argument('--model-file', default='model/model.pth',
                        help='model file')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    test_set = NMTDataset(
        src_data_path=args.test_src_data_path,
        tgt_data_path=args.test_tgt_data_path
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False
    )

    bleu = BLEU()

    model = seq2seq_trans(args.model_file)
    model.eval()
    model.to(DEVICE)

    predictions = []
    references = []

    for src, tgt in tqdm(test_loader):
        with torch.set_grad_enabled(False):
            outputs = translate(model, src, args)

        references += [tgt]
        predictions += outputs

    result = bleu.corpus_score(hypotheses=predictions, references=references)
    print(f'Corpus BLEU: {result.score}', file=sys.stderr)
