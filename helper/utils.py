import os
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


class SaveHandle(object):
    """handle the number of """

    def __init__(self, max_num):
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        if len(self.save_list) < self.max_num:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)


def compute_corpus_level_bleu_score(references, hypotheses):
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """

    if references[0][0] == '<bos>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references], [hyp.value for hyp in hypotheses])

    return bleu_score
