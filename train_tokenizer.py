import os
import argparse

from components.BPE.BPE_tokenizer import BPETokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train tokenizer')
    parser.add_argument('--train-file', help='corpus train file')
    parser.add_argument('--vocab-size', type=int, help='vocab size')
    parser.add_argument('--save-dir', help='vocab save dir')
    parser.add_argument('--filename', help='file name')
    args = parser.parse_args()

    train_file = args.train_file
    vocab_size = args.vocab_size
    save_dir = args.save_dir
    file_name = args.filename

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, file_name)

    bpe_tokenizer = BPETokenizer(train_file, vocab_size, save_file)
    bpe_tokenizer.train()
