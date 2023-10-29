import sys
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from sacrebleu.metrics import BLEU

from dataset.nmt_dataset import NMTDataset
from models.transformers.seq2seq_trans import seq2seq_trans
from models.transformers.utils import DEVICE, translate


def parse_args():
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--test-src-data-path', default='./data',
                        help='test src data file')
    parser.add_argument('--test-tgt-data-path', default='./data',
                        help='test tgt data file')
    parser.add_argument('--src-vocab-path', default='./data',
                        help='src vocab data file')
    parser.add_argument('--tgt-vocab-path', default='./data',
                        help='tgt vocab data file')
    parser.add_argument('--src-vocab-size', type=int,
                        help='src vocab data size')
    parser.add_argument('--tgt-vocab-size', type=int,
                        help='tgt vocab data size')
    parser.add_argument('--model-file', default='model/model.pth',
                        help='model file')
    parser.add_argument('--save-pred-file', default='./pred.txt',
                        help='save file')
    parser.add_argument('--beam-search', action=argparse.BooleanOptionalAction,
                        help='whether to use beam search')
    parser.add_argument('--beam-size', type=int, default=5,
                        help='beam size')
    parser.add_argument('--max-decoding-time-step', type=int, default=70,
                        help='maximum number of decoding time steps')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='beam search decoding hyperparameter')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    save_pred_file = args.save_pred_file

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

    model = seq2seq_trans(args)
    model.eval()
    model.to(DEVICE)

    predictions = []
    references = []

    for src, tgt in tqdm(test_loader):
        with torch.set_grad_enabled(False):
            outputs = translate(model, src, args)

        references += [s.replace('_', ' ') for s in tgt]
        predictions += [s.replace('_', ' ') for s in outputs]

    with open(save_pred_file, 'w', encoding='utf-8') as f:
        for p in tqdm(predictions):
            f.write(p + '\n')
        f.close()

    result = bleu.corpus_score(hypotheses=predictions, references=[references])
    print(f'Corpus BLEU: {result.score}', file=sys.stderr)
