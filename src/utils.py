#!/usr/bin/env python

import torch
import numpy as np
from attrdict import AttrDict
from scipy.linalg import block_diag
from collections import defaultdict
from attrdict import AttrDict 

from itertools import accumulate
from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
import random
from loguru import logger
import json

from src.common import WordPair
from src.preprocess import Preprocessor
from src.run_eval import Template as Run_eval

from transformers import AutoTokenizer
import traceback

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MyDataLoader:
    def __init__(self, cfg):
        preprocessor = Preprocessor(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.bert_path)
        
        self.data = preprocessor.forward()

        self.kernel = WordPair()
        self.config = cfg

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    

    def collate_fn(self, lst):
        if 'dep' in self.config.input_files:
            doc_id, speaker_ids,  input_ids, input_masks, input_segments,sentence_length, token2sents, utterance_index, \
            token_index, thread_length, token2speaker, reply_mask, speaker_mask, thread_mask, pieces2words, new2old, \
                triplets, pairs, entity_list, rel_list, polarity_list, thread_idxes,\
                merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, \
                adj_matrixes, utterance_level_reply_adj, utterance_level_speaker_adj = zip(*lst)
        else:
            doc_id, speaker_ids, input_ids, input_masks, input_segments,sentence_length, token2sents, utterance_index, \
            token_index, thread_length, token2speaker, reply_mask, speaker_mask, thread_mask, pieces2words, new2old, \
                triplets, pairs, entity_list, rel_list, polarity_list, thread_idxes,\
                merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length = zip(*lst)

        # merge_same_thread
        merged_dialog_length = list(map(len, merged_input_ids))
        max_lens = max(map(lambda line: max(map(len, line)), merged_input_ids))
        padding = lambda input_batch: [w + [self.config.pad_id] * (max_lens - len(w)) for line in input_batch for w in line]
        merged_input_ids = padding(merged_input_ids)
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for line in input_batch for w in line]
        merged_input_masks, merged_input_segments = map(padding, [merged_input_masks, merged_input_segments])

        if 'dep' in self.config.input_files:
            # DO add thread adj
            max_lens_mtx = max(map(lambda mtxs: max(map(len, mtxs)), adj_matrixes))
            adj_matrixes = [mtx for sample in adj_matrixes for mtx in sample]
            # if self.config.merged_thread == 1: assert max_lens_mtx == max_lens and len(adj_matrixes) == len(merged_input_ids)
            new_adj_matrixes = np.zeros([len(adj_matrixes), max_lens_mtx, max_lens_mtx], dtype=np.float32)
            for i in range(len(adj_matrixes)):
                new_adj_matrixes[i, :len(adj_matrixes[i]), :len(adj_matrixes[i])] = adj_matrixes[i]

            # DO add global mask
            sent_range = list(map(lambda x : list(accumulate(x)), sentence_length))
            cls_idx = list(map(lambda x : [0] + x[:-1], sent_range))
            sep_idx = list(map(lambda x : [i-1 for i in x], sent_range))
            doc_length = list(map(lambda x : sum(x), sentence_length))
            sentence_nums = list(map(len, sentence_length))
            # batch_size, max sentence num, max_doc_length, 1
            global_masks = np.zeros([len(doc_length), max(sentence_nums), max(doc_length), 1], dtype=np.float32)

            # add global mask
            for bat_idx, (sent_num, sep_i) in enumerate(zip(sentence_nums, sep_idx)):
                    for sent_idx, s_i in zip(range(sent_num), sep_i):
                        global_masks[bat_idx, sent_idx, s_i, :] = 1
            for bat_idx, (sent_num, cls_i, sep_i) in enumerate(zip(sentence_nums, cls_idx, sep_idx)):
                    for sent_idx, c_i, s_i in zip(range(sent_num), cls_i, sep_i):
                        global_masks[bat_idx, sent_idx, c_i+1:s_i, :] = 1

            # add utterance_level_reply_adj
            utterance_len = list(map(len, utterance_level_reply_adj))
            utterance_level_reply_adj_np = np.zeros((len(utterance_len), max(utterance_len), max(utterance_len)), dtype=np.float32)
            for i in range(len(utterance_len)):
                utterance_level_reply_adj_np[i, :utterance_len[i], :utterance_len[i]] = utterance_level_reply_adj[i][ :utterance_len[i], :utterance_len[i]]
            
            # add utterance_level_speaker_adj
            utterance_level_speaker_adj_np = np.zeros((len(utterance_len), max(utterance_len), max(utterance_len)), dtype=np.float32)
            for i in range(len(utterance_len)):
                utterance_level_speaker_adj_np[i, :utterance_len[i], :utterance_len[i]] = utterance_level_speaker_adj[i][ :utterance_len[i], :utterance_len[i]]
            
            # add utterance_level_mask
            utterance_level_mask = np.zeros([len(utterance_len), max(utterance_len)], dtype=np.float32)
            for i in range(len(utterance_len)):
                utterance_level_mask[i, :utterance_len[i]] = 1

        dialogue_length = list(map(len, input_ids))
        # addspeaker_ids
        max_lens = max(dialogue_length)
        padding = lambda input_batch: [line + [10] * (max_lens - len(line)) for line in input_batch]
        speaker_ids = padding(speaker_ids)

        max_lens = max(map(lambda line: max(map(len, line)), input_ids))
        input_ids = [w + [self.config.pad_id] * (max_lens - len(w)) for line in input_ids for w in line]
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for line in input_batch for w in line]
        input_masks, input_segments = map(padding, [input_masks, input_segments])

        max_lens = max(map(len, token2sents))
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for w in input_batch]
        token2sents, utterance_index, token_index, token2speaker = map(padding, [token2sents, utterance_index, token_index, token2speaker])

        padding_list = lambda input_batch :[list(map(list, w)) + [[0, 0, 0]] * (max(map(len, input_batch)) - len(w)) for w in input_batch]
        entity_lists, rel_lists, polarity_lists = map(padding_list, [entity_list, rel_list, polarity_list])

        max_tri_num = max(map(len, triplets))
        triplet_masks = [[1]*len(w) + [0] * (max_tri_num - len(w)) for w in triplets]
        triplets = [list(map(list, w)) + [[0] * 7] * (max_tri_num - len(w)) for w in triplets]

        sentence_masks = np.zeros([len(token2sents), max_lens, max_lens], dtype=int)
        for i in range(len(sentence_length)):
            masks = [np.triu(np.ones([lens, lens], dtype=int)) for lens in sentence_length[i]]
            masks = block_diag(*masks)
            sentence_masks[i, :len(masks), :len(masks)] = masks
        sentence_masks = sentence_masks.tolist()

        flatten_length = list(map(sum, sentence_length))
        cur_masks = (np.expand_dims(np.arange(max(flatten_length)), 0) < np.expand_dims(flatten_length, 1)).astype(np.int64)
        full_masks = (np.expand_dims(cur_masks, 2) * np.expand_dims(cur_masks, 1)).tolist()

        entity_matrix = self.kernel.list2rel_matrix4batch(entity_lists, max_lens)
        rel_matrix = self.kernel.list2rel_matrix4batch(rel_lists, max_lens)
        polarity_matrix = self.kernel.list2rel_matrix4batch(polarity_lists, max_lens)

        new_reply_masks = np.zeros([len(reply_mask), max_lens, max_lens], dtype=np.float32)
        for i in range(len(new_reply_masks)):
            lens = len(reply_mask[i])
            new_reply_masks[i, :lens, :lens] = reply_mask[i]

        new_speaker_masks = np.zeros([len(speaker_mask), max_lens, max_lens], dtype=np.float32)
        for i in range(len(new_speaker_masks)):
            lens = len(speaker_mask[i])
            new_speaker_masks[i, :lens, :lens] = speaker_mask[i]

        new_thread_masks = np.zeros([len(thread_mask), max_lens, max_lens], dtype=np.float32)
        for i in range(len(new_thread_masks)):
            lens = len(thread_mask[i])
            new_thread_masks[i, :lens, :lens] = thread_mask[i]

        # merged_input_segments = torch.LongTensor(merged_input_segments).to(self.config.device)
        res = {
            "doc_id": doc_id, 'speaker_ids':speaker_ids, 'input_ids':input_ids,'input_masks':input_masks, 'input_segments': input_segments, 'sentence_length':sentence_length,
            'ent_matrix': entity_matrix,   'rel_matrix': rel_matrix, 'pol_matrix': polarity_matrix,
            'sentence_masks': sentence_masks, 'full_masks': full_masks,             
            'triplets': triplets, 'triplet_masks': triplet_masks, 'pairs': pairs,
            'token2sents': token2sents, 'dialogue_length': dialogue_length,
            'utterance_index': utterance_index, 'token_index': token_index,
            'thread_lengths': thread_length, 'token2speakers': token2speaker,
            'reply_masks': new_reply_masks, 'speaker_masks': new_speaker_masks, 'thread_masks': new_thread_masks,
            'pieces2words': pieces2words, 'new2old': new2old, 'thread_idxes':thread_idxes,
            'merged_input_ids': merged_input_ids, 'merged_input_masks': merged_input_masks, 'merged_input_segments': merged_input_segments,
            'merged_sentence_length': merged_sentence_length, 'merged_dialog_length': merged_dialog_length,
        }
        if 'dep' in self.config.input_files:
            items = {"adj_matrixes": new_adj_matrixes, 'global_masks':global_masks,
            'utterance_level_reply_adj':utterance_level_reply_adj_np, 'utterance_level_speaker_adj':utterance_level_speaker_adj_np, 'utterance_level_mask': utterance_level_mask,
            }
            res.update(items)

        nocuda = ['sentence_length', 'thread_lengths', 'pairs', 'doc_id', 'pieces2words', 'new2old', 'merged_sentence_length', 'merged_dialog_length','thread_idxes','nonspeaker_token_positions']
        res = {k: v if k in nocuda else torch.tensor(v).to(self.config.device) for k, v in res.items()}
        return res
       
    def getdata(self):
        
        def load_data(mode):
            if mode not in self.config.input_files:
                return None
            return DataLoader(MyDataset(self.data[mode]), num_workers=0, worker_init_fn=self.worker_init, 
                                                shuffle= ('train' in mode),  batch_size=self.config.batch_size, collate_fn=self.collate_fn)
        
        modes = self.config.input_files.split()

        loaders = map(load_data, modes)

        line = 'polarity_dict target_dict aspect_dict opinion_dict entity_dict relation_dict'.split()
        for w, z in zip(line, self.data['label_dict']):
            self.config[w] = z

        res = (*loaders, self.config)

        return res
    
class RelationMetric:
    def __init__(self, config):
        self.clear()
        self.kernel = WordPair()
        self.predict_result  = defaultdict(list)
        self.config = config
    
    def trans2position(self, triplet, new2old, pieces2words):
        res = []
        """
        recover the position of entities in the original sentence

        new2old: transfer position from index with CLS and SEP to index without CLS and SEP
        pieces2words: transfer position from index of wordpiece to index of original words 

        Example:
        list0 (original sentence):"London is the capital of England"
        list1 (tokenized sentence): "Lon ##don is the capital of England"
        list2 (packed sentence): "[CLS] Lon #don is the capital of England [SEP]"
        predicted entity: (1, 2), denotes "Lon #don" in list2

        new2old: list2->list1
          = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, ...}
        pieces2words: list1->list0
          = {'0': 0, '1': 0, '2': 1, '3': 2, '4': 3, ...}

        input  -> entity in list2: "Lon #don" (1, 2)
        middle -> entity in list1: "Lon #don" (0, 1)
        output -> entity in list0: "London"   (0, 0)
        """

        head = lambda x : pieces2words[new2old[x]]
        tail = lambda x : pieces2words[new2old[x]]

        triplet = list(triplet)
        for s0, e0, s1, e1, s2, e2, pol in triplet:
            ns0, ns1, ns2 = head(s0), head(s1), head(s2)
            ne0, ne1, ne2 = tail(e0), tail(e1), tail(e2)
            res.append([ns0, ne0, ns1, ne1, ns2, ne2, pol])
        return res

    def trans2pair(self, pred_pairs, new2old, pieces2words):
        new_pairs = {}
        new_pos = lambda x : pieces2words[new2old[x]]
        for k, line in pred_pairs.items():
            new_line = []
            for s0, e0, s1, e1 in line:
                s0, e0, s1, e1 = new_pos(s0), new_pos(e0), new_pos(s1), new_pos(e1)
                new_line.append([s0, e0, s1, e1])
            new_pairs[k] = new_line
        return new_pairs

    def filter_entity(self, ent_list, new2old, pieces2words):
        res = []

        # If the entity is a sub-string of another entity, remove it
        # ent_list = sorted(ent_list, key=lambda x: (x[0], -x[1]))
        # ent_list = [w for i, w in enumerate(ent_list) if i == 0 or w[0] != ent_list[i-1][0]]

        for s, e, pol in ent_list:
            ns, ne = pieces2words[new2old[s]], pieces2words[new2old[e]]
            res.append([ns, ne, pol])
        return res

    def add_instance(self, data, pred_ent_matrix, pred_rel_matrix, pred_pol_matrix):
        """
        input_matrix: [B, Seq, Seq]
        pred_matrix: [B, Seq, Seq, 6]
        input_masks: [B, Seq]
        """
        pred_ent_matrix = pred_ent_matrix.argmax(-1) * data['sentence_masks']
        pred_rel_matrix = pred_rel_matrix.argmax(-1) * data['full_masks']
        pred_pol_matrix = pred_pol_matrix.argmax(-1) * data['full_masks'] 
        token2sents = data['token2sents'].tolist()
        new2old = data['new2old']
        pieces2words = data['pieces2words']
        doc_id = data['doc_id']

        pred_rel_matrix = np.array(pred_rel_matrix.tolist())
        pred_ent_matrix = np.array(pred_ent_matrix.tolist())
        pred_pol_matrix = np.array(pred_pol_matrix.tolist())

        for i in range(len(pred_ent_matrix)):
            ent_matrix, rel_matrix, pol_matrix = pred_ent_matrix[i], pred_rel_matrix[i], pred_pol_matrix[i]
            pred_triplet, pred_pairs = self.kernel.get_triplets(ent_matrix, rel_matrix, pol_matrix, token2sents[i])
            pred_ents = self.kernel.rel_matrix2list(ent_matrix)

            pred_ents = self.filter_entity(pred_ents, new2old[i], pieces2words[i])
            pred_pairs = self.trans2pair(pred_pairs, new2old[i], pieces2words[i])
            pred_triplet = self.trans2position(pred_triplet, new2old[i], pieces2words[i])

            self.predict_result[doc_id[i]].append(pred_ents)
            self.predict_result[doc_id[i]].append(pred_pairs)
            self.predict_result[doc_id[i]].append(pred_triplet)

    def clear(self):
        self.predict_result  = defaultdict(list)

    def save2file(self, gold_file, pred_file):
        # pol_dict = {"O": 0, "pos": 1, "neg": 2, "other": 3}
        pol_dict = self.config.polarity_dict
        reverse_pol_dict = {v: k for k, v in pol_dict.items()}
        reverse_ent_dict = {v: k for k, v in self.config.entity_dict.items()}
                
        gold_file = open(gold_file, 'r', encoding='utf-8')

        data = json.load(gold_file)

        res = []
        for line in data:
            doc_id, sentence = line['doc_id'], line['sentences']
            if doc_id not in self.predict_result:
                continue
            doc = ' '.join(sentence).split()
            new_triples = []

            prediction = self.predict_result[doc_id]
            entities = defaultdict(list)
            for head, tail, tp in prediction[0]:
                tp = reverse_ent_dict[tp]
                head, tail = head, tail + 1
                tp_dict = {'ENT-T': 'targets', 'ENT-A': 'aspects', 'ENT-O': 'opinions'}
                entities[tp_dict[tp]].append([head, tail])

            pairs = defaultdict(list)
            for key in ['ta', 'to', 'ao']:
                for s0, e0, s1, e1 in prediction[1][key]:
                    e0, e1 = e0 + 1, e1 + 1
                    pairs[key].append([s0, e0, s1, e1])

            new_triples = []
            for s0, e0, s1, e1, s2, e2, pol in prediction[2]:
                pol = reverse_pol_dict[pol]
                e0, e1, e2 = e0 + 1, e1 + 1, e2 + 1
                new_triples.append([s0, e0, s1, e1, s2, e2, pol, ' '.join(doc[s0:e0]), ' '.join(doc[s1:e1]), ' '.join(doc[s2:e2])])

            res.append({'doc_id': doc_id, 'triplets': new_triples, \
                        'targets': entities['targets'], 'aspects': entities['aspects'], 'opinions': entities['opinions'],\
                        'ta': pairs['ta'], 'to': pairs['to'], 'ao': pairs['ao']})
        logger.info('Save prediction results to {}'.format(pred_file))
        json.dump(res, open(pred_file, 'w', encoding='utf-8'), ensure_ascii=False)


    def compute(self, name='valid', action='eval', msg = ""):
        # action: pred, make prediction, save to file 
        # action: eval, make prediction, save to file and evaluate 

        gold_file = os.path.join(self.config.json_path, '{}.json'.format(name))
        if self.config.testset_name is not None:
            gold_file = os.path.join(self.config.json_path, '{}'.format(self.config.testset_name))
        args = AttrDict({
            'pred_file': os.path.join(self.config.target_dir, 'pred_{}_{}_{}.json'.format(self.config.lang, name, msg)),
            'gold_file': gold_file
        })
        
        self.save2file(args.gold_file, args.pred_file)
        if action == 'pred':
            return
        micro, iden, res = Run_eval(args).forward()
        self.clear()
        return micro[2], iden[2], res

def log_unhandled_exceptions(exctype, value, traceback):
    logger.exception(f"Uncaught exception: {value}", exc_info=(exctype, value, traceback))
    traceback.print_tb(traceback)


