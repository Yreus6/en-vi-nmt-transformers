import os
import argparse
from datasets import load_dataset
from underthesea import word_tokenize

from helper.utils import MultiProcessorWriter


def process_fn(idx, batch_size, data_len):
    en_batch = ''
    vi_batch = ''
    for i in range(idx, idx + batch_size):
        if i >= data_len:
            break

        prompt = dataset[phase][i]['prompt']
        prompt_ques = prompt.split(':')[0].strip()
        translation = dataset[phase][i]['translation']
        if 'English' in prompt_ques or 'Anh' in prompt_ques:
            en_sent = translation['target']
            vi_sent = translation['source']
        elif 'Vietnamese' in prompt_ques or 'Việt' in prompt_ques:
            en_sent = translation['source']
            vi_sent = translation['target']
        else:
            print(f'Error in data: {dataset[phase][i]}')
            continue

        en_batch += en_sent + '\n'
        vi_batch += word_tokenize(vi_sent, format='text') + '\n'

    return en_batch, vi_batch


def listener_fn(s, output_files):
    en_path, vi_path = output_files
    with open(en_path, 'a', encoding='utf-8') as f_en, open(vi_path, 'a', encoding='utf-8') as f_vi:
        f_en.write(s[0])
        f_en.flush()

        f_vi.write(s[1])
        f_vi.flush()


def generate_data(dataset, phase, output_path):
    en_path = os.path.join(output_path, f'{phase}.en.txt')
    vi_path = os.path.join(output_path, f'{phase}.vi.txt')

    writer = MultiProcessorWriter([en_path, vi_path], process_fn, listener_fn, len(dataset[phase]), 1000)
    writer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare translation data')
    parser.add_argument('--output-path', help='output path')
    args = parser.parse_args()

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Load MTet dataset
    dataset = load_dataset('phongmt184172/mtet')

    for phase in ['train', 'validation', 'test']:
        print(f'Generate {phase} data...')
        phase_path = os.path.join(output_path, phase)
        if not os.path.exists(phase_path):
            os.mkdir(phase_path)
        generate_data(dataset, phase, phase_path)
