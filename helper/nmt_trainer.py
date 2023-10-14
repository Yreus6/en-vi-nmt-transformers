import os
import sys
import numpy as np
import torch
from torch import optim, nn
import logging
import time

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.loading_data import loading_data
from models.transformers.seq2seq_trans import (
    Seq2SeqTransformer,
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    EMB_SIZE,
    NHEAD,
    FFN_HID_DIM,
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE
)
from models.transformers.utils import DEVICE, create_mask, PAD_IDX
from .trainer import Trainer
from .utils import SaveHandle

torch.manual_seed(42)


class NMTTrainer(Trainer):
    def setup(self):
        args = self.args
        self.model = Seq2SeqTransformer(
            NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM
        )
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.model.to(DEVICE)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_loader, val_loader = loading_data(args)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.start_epoch = 1
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, DEVICE)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, DEVICE))
            else:
                raise RuntimeError('Invalid model file')

        self.log_dir = os.path.join('./runs', args.save_dir.split('/')[-1])
        self.writer = SummaryWriter(self.log_dir)

        self.save_list = SaveHandle(max_num=args.max_model_num)

        self.hist_valid_scores = []

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self):
        epoch_start = time.time()
        self.model.train()

        losses = 0.

        for src, tgt in tqdm(self.train_loader):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = self.model(
                src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask
            )

            self.optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            # clip gradient
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip_grad)

            self.optimizer.step()
            losses += loss.item()

        train_loss = losses / len(list(self.train_loader))

        logging.info(
            'Epoch {} Train, Loss: {:.2f} Cost {:.1f} sec'.format(self.epoch, train_loss, time.time() - epoch_start)
        )
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()

        losses = 0.
        cum_tgt_words = 0.

        for src, tgt in tqdm(self.val_loader):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            with torch.set_grad_enabled(False):
                logits = self.model(
                    src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask
                )

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
            tgt_word_num_to_predict = sum(s.shape[0] for s in tgt.transpose(0, 1))
            cum_tgt_words += tgt_word_num_to_predict

        val_loss = losses / len(list(self.val_loader))
        ppl = np.exp(losses / cum_tgt_words)
        self.writer.add_scalar('val/ppl', ppl, self.epoch)

        logging.info(
            'Epoch {} Val, Loss: {:.2f} PPL: {:.2f} Cost {:.1f} sec'
            .format(self.epoch, val_loss, ppl, time.time() - epoch_start)
        )

        valid_metric = -ppl
        is_better = len(self.hist_valid_scores) == 0 or valid_metric > max(self.hist_valid_scores)
        self.hist_valid_scores.append(valid_metric)

        model_state_dic = self.model.state_dict()

        if is_better:
            print('save currently the best model to [%s]' % self.save_dir, file=sys.stderr)
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
