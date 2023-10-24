import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
from underthesea import word_tokenize


def generate_data(dataset, phase, output_path):
    en_path = os.path.join(output_path, f'{phase}.en.txt')
    vi_path = os.path.join(output_path, f'{phase}.vi.txt')

    with open(en_path, 'w', encoding='utf-8') as f_en, open(vi_path, 'w', encoding='utf-8') as f_vi:
        for i in tqdm(range(len(dataset[phase]))):
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

            f_en.write(en_sent + '\n')
            f_vi.write(word_tokenize(vi_sent, format='text') + '\n')

        f_en.close()
        f_vi.close()


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
