import os
import argparse
from underthesea import word_tokenize

from helper.utils import MultiProcessorWriter


def process_fn(idx, batch_size, data_len):
    vi_batch = ''
    for i in range(idx, idx + batch_size):
        if i >= data_len:
            break

        vi_batch += word_tokenize(lines[i], format='text') + '\n'

    return vi_batch


def listener_fn(s, output_files):
    vi_path = output_files[0]
    with open(vi_path, 'a', encoding='utf-8') as f_vi:
        f_vi.write(s)
        f_vi.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare translation data')
    parser.add_argument('--src-vi-path', help='src vi path')
    parser.add_argument('--output-path', help='output path')
    parser.add_argument('--filename', help='output filename')
    args = parser.parse_args()

    src_vi_path = args.src_vi_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_vi_file = os.path.join(output_path, args.filename)

    with open(src_vi_path, encoding='utf-8') as f:
        lines = f.readlines()

    writer = MultiProcessorWriter([output_vi_file], process_fn, listener_fn, len(lines), 1000)
    writer.run()
