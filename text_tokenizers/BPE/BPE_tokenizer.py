from tokenizers import (
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
    processors
)

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class BPETokenizer:
    def __init__(self, corpus_src, vocab_size, save_file):
        self.tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_symbols)
        self.corpus_src = corpus_src
        self.save_file = save_file

    def get_training_corpus(self):
        with open(self.corpus_src, encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 1000):
                yield lines[i: i + 1000]

    def train(self):
        self.tokenizer.train_from_iterator(self.get_training_corpus(), trainer=self.trainer)
        self.tokenizer.save(path=self.save_file)

    @staticmethod
    def get_tokenizer(vocab_file):
        tokenizer = Tokenizer.from_file(vocab_file)
        tokenizer.post_processor = processors.TemplateProcessing(
            single='<bos> $A <eos>',
            pair='<bos>:0 $A:0 <eos>:0 $B:1 <eos>:1',
            special_tokens=[('<bos>', 2), ('<eos>', 3)]
        )

        return tokenizer
