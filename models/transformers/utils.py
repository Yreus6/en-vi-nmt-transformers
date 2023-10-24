import torch
from tokenizers import decoders

from text_tokenizers.BPE.BPE_tokenizer import BPETokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break

    return ys


def translate(model, src_sentence_batch, args):
    src_tokenizer = BPETokenizer.get_tokenizer(args.src_vocab_path)
    tgt_tokenizer = BPETokenizer.get_tokenizer(args.tgt_vocab_path)
    decoder = decoders.ByteLevel()
    encoded_src_batch = src_tokenizer.encode_batch(src_sentence_batch)
    tgt_batch = []

    for i in range(encoded_src_batch):
        src_token_ids = encoded_src_batch[i].ids
        src = torch.tensor(src_token_ids)
        num_token_ids = len(src_token_ids)
        src_mask = (torch.zeros(num_token_ids, num_token_ids)).type(torch.bool)
        tgt_token_ids = greedy_decode(model, src, src_mask, max_len=num_token_ids + 5, start_symbol=BOS_IDX).flatten()
        tgt_tokens = [tgt_tokenizer.id_to_token(x) for x in tgt_token_ids if (x != BOS_IDX and x != EOS_IDX)]
        tgt_batch.append(tgt_tokens)

    return [decoder.decode(x) for x in tgt_batch]
