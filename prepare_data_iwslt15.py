import os
import argparse
from underthesea import word_tokenize

from helper.utils import MultiProcessorWriter


def process_fn(idx, batch_size, data_len):
    en_batch = ''
    vi_batch = ''
    for i in range(idx, idx + batch_size):
        if i >= data_len:
            break

        en_sent, vi_sent = lines[i]

        en_batch += en_sent.rstrip('\n') + '\n'
        vi_batch += word_tokenize(vi_sent.rstrip('\n'), format='text') + '\n'

    return en_batch, vi_batch


def listener_fn(s, output_files):
    en_path, vi_path = output_files
    with open(en_path, 'a', encoding='utf-8') as f_en, open(vi_path, 'a', encoding='utf-8') as f_vi:
        f_en.write(s[0])
        f_en.flush()

        f_vi.write(s[1])
        f_vi.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare translation data')
    parser.add_argument('--src-path', help='src path')
    parser.add_argument('--tgt-path', help='tgt path')
    parser.add_argument('--output-path', help='output path')
    parser.add_argument('--src-filename', help='output src filename')
    parser.add_argument('--tgt-filename', help='output tgt filename')
    args = parser.parse_args()

    src_path = args.src_path
    tgt_path = args.tgt_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_src_file = os.path.join(output_path, args.src_filename)
    output_tgt_file = os.path.join(output_path, args.tgt_filename)

    with open(src_path, encoding='utf-8') as f_src, open(tgt_path, encoding='utf-8') as f_tgt:
        lines = list(zip(f_src.readlines(), f_tgt.readlines()))

    writer = MultiProcessorWriter([output_src_file, output_tgt_file], process_fn, listener_fn, len(lines), 1000)
    writer.run()
