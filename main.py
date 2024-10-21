#!/usr/bin/env python


import os
import subprocess
import yaml
import argparse

import torch
import sys
import numpy as np

from attrdict import AttrDict
from torch.optim import AdamW
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

from src.utils import MyDataLoader, RelationMetric 
from src.model import BertWordPair
from src.common import set_seed, ScoreManager, update_config
from tqdm import tqdm
from loguru import logger
from src.utils import log_unhandled_exceptions
import datetime
import csv

sys.excepthook = log_unhandled_exceptions


class Main:
    def __init__(self, args):
        config = AttrDict(yaml.load(open('src/config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))

        for k, v in vars(args).items():
            setattr(config, k, v)
        
        config = update_config(config)

        set_seed(config.seed)
        if not os.path.exists(config.target_dir):
            os.makedirs(config.target_dir)

        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        # config.device = torch.device('cpu')
        self.config = config


    def train_iter(self):
        train_data = tqdm(self.trainLoader, total=self.trainLoader.__len__(), file=sys.stdout, dynamic_ncols=True)
        losses = []

        for i, data in enumerate(train_data):
            self.model.train()
            
            loss, loss_list, _ = self.model(**data)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            
            loss_list = loss_list
            losses.append([float(w) for w in loss_list])

            train_losses = np.mean(losses, 0)
            description = "Epoch {}, Loss (entity:{:.4f}, rel: {:.4f}, pol: {:.4f})".format(self.global_epoch, *train_losses)
            
            train_data.set_description(description)
            # if i==200: break

        logger.info(description)

        return train_losses

    def evaluate_iter(self, dataLoader=None):

        self.model.eval()
        dataLoader = self.validLoader if dataLoader is None else dataLoader
        dataiter = tqdm(dataLoader, total=dataLoader.__len__(), file=sys.stdout, dynamic_ncols=True)
        losses = []
        with torch.no_grad():
            for i, data in enumerate(dataiter):
                _, loss_list, (pred_ent_matrix, pred_rel_matrix, pred_pol_matrix) = self.model(**data)
                self.relation_metric.add_instance(data, pred_ent_matrix, pred_rel_matrix, pred_pol_matrix)

                losses.append( [float(w) for w in loss_list] )
            
            eval_losses = np.mean(losses, 0)
            logger.info("EvalLoss (entity:{:.4f}, rel: {:.4f}, pol: {:.4f})".format(*eval_losses))
            
            for i, data in enumerate(dataiter):
                continue
            return eval_losses
 
    def evaluate(self, epoch=0, action='eval'):
        PATH = os.path.join(self.config.target_dir, "{}_{}.pth.tar").format(self.config.lang, self.time)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        self.model.eval()

        self.evaluate_iter(self.testLoader)
        result = self.relation_metric.compute('test', action, msg ="f1_{:.4f}ident_{:.4f}epo_{}".format(self.best_score*100, self.best_ident*100, epoch))

        if action == 'eval':
            self.test_score, self.test_ident, self.test_res = result
            logger.info(self.test_res)
            logger.info("Evaluate on test set, micro-F1 score: {:.4f}%, ident: {:.4f}%".format(self.test_score * 100, self.test_ident * 100))

    def train(self):
        best_score, best_ident, best_iter = 0, 0, 0
        best_testset_score, best_testset_ident = -1, -1
        for epoch in range(self.config.epoch_size):
            self.global_epoch = epoch
            train_losses = self.train_iter()
            
            eval_losses = self.evaluate_iter()
            score, ident, res = self.relation_metric.compute(msg=f"{self.time}")
            self.score_manager.add_instance(score, res)
            logger.info("Epoch {}, validset micro-F1 score: {:.4f}%, ident-f1 score: {:.4f}%".format(epoch, score * 100, ident * 100))
            logger.info(res)
                
            # early stopping when training loss tends to converge,  avoiding overfitting
            if sum(train_losses) < 1e-3:
                if score > best_score:
                    best_score, best_ident, best_iter = score, ident, epoch

                    torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict(), 'best_score': best_score, 'config': self.config},
                            os.path.join(self.config.target_dir,  "{}_{}.pth.tar".format(self.config.lang, self.time)))
                    self.model.to(self.config.device)
                    logger.info("This is a new best epoch in validset!")
                elif epoch - best_iter > self.config.patience:
                    print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                    break
            self.model.to(self.config.device)
            
            logger.info("\n")
            # if epoch == 1: break
        
        self.best_iter = best_iter
        self.best_score = best_score
        self.best_ident = best_ident
    
    def check_param(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        is_bert = ['bert']

        bert_decay = [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and any(nd in n for nd in is_bert)]
        bert_nodecay = [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and any(nd in n for nd in is_bert)]
        other_decay = [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(nd in n for nd in is_bert)]
        other_nodecay = [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not any(nd in n for nd in is_bert)]
        a = set(bert_decay + bert_nodecay + other_decay + other_nodecay)
        b = set([p for n, p in param_optimizer])
        print("is params all be add", a==b)

    def load_param(self):
        self.check_param()
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        is_bert = ['bert']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and any(nd in n for nd in is_bert)], 'weight_decay': self.config.weight_decay, 'lr':self.config.bert_lr}, 
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and any(nd in n for nd in is_bert)], 'weight_decay': 0, 'lr':self.config.bert_lr}, 
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(nd in n for nd in is_bert)], 'weight_decay': self.config.weight_decay, 'lr':self.config.lr},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not any(nd in n for nd in is_bert)], 'weight_decay': 0, 'lr':self.config.lr}]

        
        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=float(self.config.bert_lr),
                               eps=float(self.config.adam_epsilon))

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=self.config.epoch_size * self.trainLoader.__len__())

    def forward(self):
        self.trainLoader, self.validLoader, self.testLoader, config = MyDataLoader(self.config).getdata()
        self.model = BertWordPair(self.config).to(config.device)
        self.score_manager = ScoreManager()
        self.relation_metric = RelationMetric(self.config)
        self.load_param()


        self.time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_name_with_time = './result/{}/{}.log'.format(self.config.lang, self.time)
        logger.add(log_name_with_time, level="INFO")

        # write config to log
        for k, v in (self.config).items():
            logger.info("{}:\t{}".format(k, v))
        logger.info("Start training...")

        # train and evaluate
        self.train()
        logger.info("Training finished..., best epoch is {}...".format(self.best_iter))
        if 'test' in self.config.input_files:
            logger.info("Start testing...")
            self.evaluate(self.best_iter, action='eval')

        # rename the log file
        new_name = "./result/{}/f1_{:.4f}ident_{:.4f}ep_{}.log".format(self.config.lang,self.test_score*100, self.test_ident*100, self.best_iter)
        os.rename(log_name_with_time, new_name)

        # save the result to csv file
        file_path = './result/{}/{}.csv'.format(self.config.lang, self.config.result_file_name)
        with open(file_path, 'a', newline='') as csvfile:
            fieldnames = [ 'avg', 'f1', 'ident', 'ta', 'to', 'ao', 't', 'a', 'o', 'intra', 'inter', 'cross-1', 'cross-ge2', 'cross-ge3', 'val_f1', 'val_ident', 'epoch', 'dropout', 'batch_size',  'logger', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
        
        # parse res
        res = self.test_res.split('------------------------------')
        def get_f1(idx1, idx2):
            return "{:.4f}".format((float(res[idx1].split('\n')[idx2].split('\t')[3])))
        t, a, o = get_f1(1, 1), get_f1(1, 2), get_f1(1, 3)
        ta, to, ao = get_f1(2, 1), get_f1(2, 2), get_f1(2, 3)
        inter, intra, cross1, cross2, cross3 = get_f1(4, 1), get_f1(4, 2), get_f1(4, 3), get_f1(4, 4), get_f1(4, 5)

        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["{:.4f}".format((self.test_score+self.test_ident)/2*100), 
            "{:.4f}".format(self.test_score*100), "{:.4f}".format(self.test_ident*100),
            ta, to, ao, t, a, o, inter, intra, cross1, cross2, cross3,
                "{:.4f}".format(self.best_score*100), "{:.4f}".format(self.best_ident*100), self.best_iter, 
            self.config.dropout, self.config.batch_size,  new_name, self.time])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', type=str, default='en', choices=['zh', 'en'], help='language selection')
    parser.add_argument('-c', '--cuda_index', type=int, default=0, help='CUDA index')
    parser.add_argument('-i', '--input_files', type=str, default='train valid test', help='input file names')
    parser.add_argument('-a', '--action', type=str, default='train', choices=['train', 'eval', 'pred'], help='choose to train, evaluate, or predict')
    parser.add_argument('-b', '--best_iter', type=int, default=0, help='best iter to run test, only used when action is eval or pred')
    parser.add_argument('-s', '--seed', type=int, default=45, help='random seed')
    parser.add_argument('--result_file_name', type=str, default='result')

    parser.add_argument('--adam_epsilon', type=float, default=1e-7)
    parser.add_argument('--warmup_steps', type=int, default=350)
    
    parser.add_argument('--dscgnn_layer_num', type=int, default=2)
    parser.add_argument('--gnn_layer_num', type=int, default=3)
    parser.add_argument('--gnn_dropout', type=float, default=0.1)

    parser.add_argument('--merged_thread', type=int, default=1, help='whether encoding by thread')
    parser.add_argument('--root_merge', type=int, default=1, help='whether to copy root for each thread')

    parser.add_argument('--loss_w', type=str, default="296")
    parser.add_argument('--topk', type=float, default=0.8)

    parser.add_argument('--testset_name', type=str, default=None, help='path to trainset')
    args = parser.parse_args()
    main = Main(args)
    main.forward()
