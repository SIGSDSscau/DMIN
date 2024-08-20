#!/usr/bin/env python
# _*_ coding:utf-8 _*_


from src.Roberta import MultiHeadAttention,  InteractionAttention
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from itertools import accumulate

import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1, d_out=None):
        super(PositionwiseFeedForward, self).__init__()
        if d_out is None: d_out = d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class InteractLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, config=None):
        super(InteractLayer, self).__init__()
        head_size = int(d_model / num_heads)
        self.config = config
        self.interactionAttention = InteractionAttention(num_heads, d_model, head_size, head_size, dropout, config=config)

        self.layer_norm_pre = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.layer_norm_post = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, global_x, mask, sentence_length,):
        x = self.layer_norm_pre(self.interactionAttention(x, global_x, mask,)[0] + x)
        x = self.layer_norm_post(self.ffn(x) + x)
        x = self.dropout(x)
        return x


class GCN(nn.Module):
    def __init__(self, config, layer_num, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.config = config
        self.layer_list = nn.ModuleList()
        for i in range(layer_num):
            if i == layer_num - 1:
                self.layer_list.append(nn.Linear(hidden_dim, output_dim))
            elif i == 0:
                self.layer_list.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layer_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.gnn_dropout = nn.Dropout(dropout)
        self.gnn_activation = F.gelu

    def forward(self, x, mask, adj):
        D_hat = torch.diag_embed(torch.pow(torch.sum(adj, dim=-1), -1))
        if torch.isinf(D_hat).any():
            D_hat[torch.isinf(D_hat)] = 0.0
        adj = torch.matmul(D_hat, adj)
        # adj = torch.matmul(adj, D_hat)

        x_mask = mask.unsqueeze(-1)#.expand(-1, -1, x.size(-1))
        for i, layer in enumerate(self.layer_list):
            if i != 0:
                x = self.gnn_dropout(x)
            x = torch.matmul(x, layer.weight.T) + layer.bias
            x = torch.matmul(adj, x)
            x = x * x_mask
            x = self.gnn_activation(x)

        return x


class BertWordPair(nn.Module):
    def __init__(self, config):
        super(BertWordPair, self).__init__()
        self.config = config 
        self.bert = AutoModel.from_pretrained(config.bert_path)
        
        bert_config = AutoConfig.from_pretrained(config.bert_path)
        bh = bert_config.hidden_size
        nhead = bert_config.num_attention_heads
        att_head_size = int(bh / nhead)

        self.config.loss_weight = {'ent': int(self.config.loss_w[0]), 'rel': int(self.config.loss_w[1]), 'pol': int(self.config.loss_w[2])}
        
        self.inner_dim = 256
        self.ent_dim = self.inner_dim * 4 * 4
        self.rel_dim = self.inner_dim * 4 * 3
        self.pol_dim = self.inner_dim * 4 * 4

        self.dense_all = nn.Linear(bert_config.hidden_size, self.ent_dim+self.rel_dim+self.pol_dim)
        
        self.dropout = nn.Dropout(config.dropout)


        self.interactLayer = InteractLayer(
            bert_config.hidden_size,
            bert_config.num_attention_heads,
            bert_config.hidden_dropout_prob,
            config
        )
                
        self.layernorm = nn.LayerNorm(bh, eps=1e-12)
        self.syngcn = GCN(config, config.gnn_layer_num, bh, bh, bh, config.gnn_dropout)

        self.semgcn = GCN(config, config.gnn_layer_num, bh, bh, bh, config.gnn_dropout)
        self.semantic_attention = MultiHeadAttention(bert_config.num_attention_heads, bh, att_head_size, att_head_size, bert_config.attention_probs_dropout_prob)

        # topk
        self.topK_select_layer = nn.Linear(bh, 1)
        self.utt_linear = nn.Linear(3*bh, bh)
        
        self.dscgcn = GCN(config, config.dscgnn_layer_num, bh, bh, bh, config.gnn_dropout)
        self.global_layernorm = nn.LayerNorm(bh, eps=1e-12)

    def custom_sinusoidal_position_embedding(self, token_index, pos_type):
        """
        See RoPE paper: https://arxiv.org/abs/2104.09864
        https://blog.csdn.net/weixin_43646592/article/details/130924280
        """
        output_dim = self.inner_dim
        position_ids = token_index.unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(self.config.device) # 128
        if pos_type == 0:
            indices = torch.pow(10000, -2 * indices / output_dim)
        else:
            indices = torch.pow(15, -2 * indices / output_dim)
        embeddings = position_ids * indices # [seq_len, 128]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1) # [seq_len, 128, 2]
        embeddings = embeddings.repeat((1, *([1]*len(embeddings.shape)))) # [1, seq_len, 128, 2]
        embeddings = torch.reshape(embeddings, (1, len(token_index), output_dim)) # [1, seq_len, 256]
        embeddings = embeddings.squeeze(0)
        return embeddings
    
    def get_instance_embedding(self, qw: torch.Tensor, kw: torch.Tensor, token_index, thread_length, pos_type):
        """_summary_
        Parameters
        ----------
        qw : torch.Tensor, (seq_len, class_nums, hidden_size)
        kw : torch.Tensor, (seq_len, class_nums, hidden_size)
        token_index : 相对根的线程token位置
        """

        seq_len, num_classes = qw.shape[:2]

        accu_index = [0] + list(accumulate(thread_length))

        logits = qw.new_zeros([seq_len, seq_len, num_classes])

        # Compute the ROPE matrix
        for i in range(len(thread_length)):
            for j in range(len(thread_length)):
                rstart, rend = accu_index[i], accu_index[i+1]
                cstart, cend = accu_index[j], accu_index[j+1]

                cur_qw, cur_kw = qw[rstart:rend], kw[cstart:cend]
                x, y = token_index[rstart:rend], token_index[cstart:cend]

                # This is used to compute relative distance, see the matrix in Fig.8 of our paper
                x = - x if i > 0 and i < j else x
                y = - y if j > 0 and i > j else y

                x_pos_emb = self.custom_sinusoidal_position_embedding(x, pos_type) # 38，256（38是第一个句子的长度
                y_pos_emb = self.custom_sinusoidal_position_embedding(y, pos_type)

                # Refer to https://kexue.fm/archives/8265
                x_cos_pos = x_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1) # 38， 1， 256
                x_sin_pos = x_pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
                cur_qw2 = torch.stack([-cur_qw[..., 1::2], cur_qw[..., ::2]], -1) # [38, 6, 128, 2]
                cur_qw2 = cur_qw2.reshape(cur_qw.shape) #[38, 6, 256]
                cur_qw = cur_qw * x_cos_pos + cur_qw2 * x_sin_pos #[38, 6, 256]

                y_cos_pos = y_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                y_sin_pos = y_pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
                cur_kw2 = torch.stack([-cur_kw[..., 1::2], cur_kw[..., ::2]], -1)
                cur_kw2 = cur_kw2.reshape(cur_kw.shape)
                cur_kw = cur_kw * y_cos_pos + cur_kw2 * y_sin_pos

                pred_logits = torch.einsum('mhd,nhd->mnh', cur_qw, cur_kw).contiguous() #38 38 6， 38 34 6
                logits[rstart:rend, cstart:cend] = pred_logits #[0:38, 0:38]=[38,38,6] [0:38, 108:142]=[38,34,6] [38:108, 108:142]=[70,34,6]

        return logits 

    def get_ro_embedding(self, qw, kw, token_index, thread_lengths, pos_type):
        # qw_res = qw.new_zeros(*qw.shape)
        # kw_res = kw.new_zeros(*kw.shape)
        # qw,kw (batch_size, seq_len, 6, 256)
        logits = []
        batch_size = qw.shape[0]
        for i in range(batch_size):
            pred_logits = self.get_instance_embedding(qw[i], kw[i], token_index[i], thread_lengths[i], pos_type) #[seqlen, seqlen, classnums]
            logits.append(pred_logits)
        logits = torch.stack(logits)
        return logits 


    def classify_matrix(self, kwargs, sequence_outputs, input_labels, masks, mat_name='ent'):

        utterance_index, token_index, thread_lengths = [kwargs[w] for w in ['utterance_index', 'token_index', 'thread_lengths']]

        outputs = torch.split(sequence_outputs, self.inner_dim * 4, dim=-1) # (batch_size, seq_len, 256*4) * 6
        outputs = torch.stack(outputs, dim=-2) # (batch_size, seq_len, 6, 256*4)

        q_token, q_utterance, k_token, k_utterance = torch.split(outputs, self.inner_dim, dim=-1) # (batch_size, seq_len, 6, 256)


        if self.config.use_rope == True:
            if mat_name == 'ent':
                # [batch_size, seq_len, seq_len, class_nums]
                pred_logits = self.get_ro_embedding(q_token, k_token, token_index, thread_lengths, pos_type=0) # pos_type=0 for token-level relative distance encoding
            else:
                pred_logits0 = self.get_ro_embedding(q_token, k_token, token_index, thread_lengths, pos_type=0)
                pred_logits1 = self.get_ro_embedding(q_utterance, k_utterance, utterance_index, thread_lengths, pos_type=1) # pos_type=1 for utterance-level relative distance encoding
                pred_logits = pred_logits0 + pred_logits1
        else:
            # without rope, use dot-product attention directly
            pred_logits = torch.einsum('bmhd,bnhd->bmnh', q_token, k_token).contiguous()

        nums = pred_logits.shape[-1]

        # alpha^k = loss weight
        criterion = nn.CrossEntropyLoss(sequence_outputs.new_tensor([1.0] + [self.config.loss_weight[mat_name]] * (nums - 1)))
            

        active_loss = masks.view(-1) == 1
        active_logits = pred_logits.view(-1, pred_logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]
        
        loss = criterion(active_logits, active_labels)
        
        return loss, pred_logits 

    
    def merge_sentence(self, sequence_outputs, input_masks, dialogue_length):
        res = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            for j in range(s, e):
                lens = input_masks[j].sum()
                stack.append(sequence_outputs[j, :lens])
            res.append(torch.cat(stack))           
        new_res = sequence_outputs.new_zeros([len(res), max(map(len, res)), sequence_outputs.shape[-1]])
        for i, w in enumerate(res):
            new_res[i, :len(w)] = w
        return new_res # batch_size, max_dialogue_length, hidden_size
    

    
    def root_merge_sentence(self, sequence_outputs, input_masks, dialogue_length, thread_lengths):
        if self.config.root_merge == 0:
            return self.merge_sentence(sequence_outputs, input_masks, dialogue_length)
        
        res = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            root_stack = []
            root_len = thread_lengths[i][0]
            for j in range(s, e):
                lens = input_masks[j].sum()
                root_stack.append(sequence_outputs[j, :root_len])
                stack.append(sequence_outputs[j, root_len:lens])

            root = torch.stack(root_stack).sum(0) / len(root_stack)

            stack = [root] + stack
            res.append(torch.cat(stack))  
        new_res = sequence_outputs.new_zeros([len(res), max(map(len, res)), sequence_outputs.shape[-1]])
        for i, w in enumerate(res):
            new_res[i, :len(w)] = w
        return new_res # batch_size, max_dialogue_length, hidden_size
    

    def topk_aggregate(self, sentence_sequence_outputs, global_masks):
        batch_size, max_dialogue_length, hidden_size = sentence_sequence_outputs.shape
        batch_size, max_sentence_num, max_dialogue_length, _ = global_masks.shape

        sentence_lengths = global_masks.sum(dim=2).squeeze(-1) 

        split_sentences = []

        for i in range(batch_size):
            split_sentences.append([])
            for j in range(max_sentence_num):
                sentence_length = sentence_lengths[i, j]
                if sentence_length > 0:
                    start_index = (global_masks[i, j, :, :] == 1).nonzero()[0, 0].item()
                    end_index = int(start_index + sentence_length.item())
                    token_representation = sentence_sequence_outputs[i, start_index:end_index-1, :]
                    speaker_representation = sentence_sequence_outputs[i, end_index-1, :]
                    score = self.topK_select_layer(token_representation).squeeze(-1) / (sentence_length - 1)
                    # get topk of sentence: pooling[avg, max, speaker] as sentence representation
                    k = int(self.config.topk*sentence_length)
                    k = k if k > 0 else 1

                    topk = torch.topk(score, k, dim=0, largest=True)[1]
                    score = torch.softmax(score[topk], dim=0)
                    token_representation = token_representation[topk]

                    token_representation = token_representation * score.unsqueeze(-1)

                    utt_representation = self.utt_linear(torch.cat((token_representation.mean(dim=0), token_representation.max(dim=0)[0], speaker_representation), dim=-1))
                     
                    split_sentences[i].append(utt_representation)
                else:
                    split_sentences[i].append(sentence_sequence_outputs.new_zeros([hidden_size]))
        split_sentences = torch.stack([torch.stack(bat) for bat in split_sentences], dim=0)

        return split_sentences
    
    def global_encoding(self, speaker_ids, sentence_sequence_outputs, global_masks, utterance_level_reply_adj, utterance_level_speaker_adj, utterance_level_mask):
        # sentence_sequence_outputs: batch_size, max_dialogue_length, hidden_size
        # global_masks: batch_size, max_sentence_num, max_dialogue_length, 1
        
        utterance_sequence = self.topk_aggregate(sentence_sequence_outputs, global_masks)      
        global_outputs = self.dscgcn(utterance_sequence, utterance_level_mask, utterance_level_reply_adj)
        global_outputs = self.global_layernorm(utterance_sequence + global_outputs)          

        return global_outputs
    
    def utterance2thread(self, sequence_outputs, thread_idxes, sentence_length, thread_lengths, merged_input_masks):
        # sequence_outputs: batch_size, max_sentence_length, hidden_size
        thread_num, max_thread_len = merged_input_masks.shape
        
        thread_sequence_output = sequence_outputs.new_zeros([thread_num, max_thread_len, sequence_outputs.shape[-1]])
        thread_idx = 0
        for bat_idx, bat in enumerate(thread_idxes):
            for t_idx, thread in enumerate(bat):
                thread_list = []
                for s_idx, sent_idx in enumerate(thread):
                    thread_list.append(sequence_outputs[bat_idx, :sentence_length[bat_idx][sent_idx],:])
                thread_list = torch.cat(thread_list, dim=0)
                thread_sequence_output[thread_idx, :thread_list.shape[0], :] = thread_list
                thread_idx += 1
        
        return thread_sequence_output
    


    def forward(self, **kwargs):
        if self.config.merged_thread == 0:
            input_ids, input_masks, input_segments = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments']]

        sentence_length, thread_idxes, merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, merged_dialog_length, thread_lengths, adj_matrixes \
            = [kwargs[w] for w in ['sentence_length', 'thread_idxes', 'merged_input_ids', 'merged_input_masks', 'merged_input_segments', 'merged_sentence_length', 'merged_dialog_length', 'thread_lengths',  'adj_matrixes', ]]
        
        ent_matrix, rel_matrix, pol_matrix = [kwargs[w] for w in ['ent_matrix', 'rel_matrix', 'pol_matrix']]
        reply_masks, speaker_masks, thread_masks = [kwargs[w] for w in ['reply_masks', 'speaker_masks', 'thread_masks']]
        sentence_masks, full_masks, dialogue_length = [kwargs[w] for w in ['sentence_masks', 'full_masks', 'dialogue_length']]
        
        # DO 
        # 1. bert encoding
        if self.config.merged_thread == 1: 
            sequence_outputs = self.bert(merged_input_ids, token_type_ids=merged_input_segments, attention_mask=merged_input_masks)[0] # utterance_num, seq_len, hidden_size
            sentence_sequence_outputs = self.root_merge_sentence(sequence_outputs, merged_input_masks, merged_dialog_length, thread_lengths)
        else: # w/o thread
            sequence_outputs = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0] # utterance_num, seq_len, hidden_size
            sentence_sequence_outputs = self.merge_sentence(sequence_outputs, input_masks, dialogue_length)

        sentence_sequence_outputs = self.dropout(sentence_sequence_outputs)

        # 2. local encoding 
        # 2.1 add thread gcn syntactic
        if self.config.merged_thread == 1:
            syngcn_outputs = self.syngcn(sequence_outputs, merged_input_masks, adj_matrixes)
            syngcn_outputs = self.root_merge_sentence(syngcn_outputs, merged_input_masks, merged_dialog_length, thread_lengths)
        else: # w/o thread
            syngcn_outputs = self.syngcn(sequence_outputs, input_masks, adj_matrixes)
            syngcn_outputs = self.merge_sentence(syngcn_outputs, input_masks, dialogue_length)
        
        syngcn_outputs = self.dropout(syngcn_outputs)


        # 2.2 add thread gcn semantic
        _, semantic_adj = self.semantic_attention(sequence_outputs, sequence_outputs, sequence_outputs)
        semantic_adj = semantic_adj.mean(dim=1)
        if self.config.merged_thread == 1:
            semgcn_output = self.semgcn(sequence_outputs, merged_input_masks, semantic_adj)
            semgcn_output = self.root_merge_sentence(semgcn_output, merged_input_masks, merged_dialog_length, thread_lengths)
        else: # w/o thread
            semgcn_output = self.semgcn(sequence_outputs, input_masks, semantic_adj)
            semgcn_output = self.merge_sentence(semgcn_output, input_masks, dialogue_length)
        semgcn_output = self.dropout(semgcn_output)
   
        # 2.3 integrate syntactic and semantic
        sequence_outputs = self.layernorm(sentence_sequence_outputs+syngcn_outputs+semgcn_output)
        
        # 3. global encoding 
        global_masks, utterance_level_reply_adj, utterance_level_speaker_adj, utterance_level_mask, speaker_ids =  [kwargs[w] for w in ['global_masks', 'utterance_level_reply_adj', 'utterance_level_speaker_adj', 'utterance_level_mask','speaker_ids']]
        global_outputs = self.global_encoding(speaker_ids, sentence_sequence_outputs, global_masks, utterance_level_reply_adj, utterance_level_speaker_adj, utterance_level_mask) 
        
        # 4. Interaction attention
        thread_masks = thread_masks.bool().unsqueeze(1)
        sequence_outputs = self.interactLayer(sequence_outputs, global_outputs, thread_masks, sentence_length=sentence_length,)

        # 5. decode
        sequence_outputs = self.dense_all(sequence_outputs)
        sequence_ent = sequence_outputs[:, :, :self.ent_dim]
        sequence_rel = sequence_outputs[:, :, self.ent_dim:self.ent_dim + self.rel_dim]
        sequence_pol = sequence_outputs[:, :, self.ent_dim + self.rel_dim:]
    
        ent_loss, ent_logit = self.classify_matrix(kwargs, sequence_ent, ent_matrix, sentence_masks, 'ent')
        rel_loss, rel_logit = self.classify_matrix(kwargs, sequence_rel, rel_matrix, full_masks, 'rel')
        pol_loss, pol_logit = self.classify_matrix(kwargs, sequence_pol, pol_matrix, full_masks, 'pol')

        total_loss = ent_loss + rel_loss + pol_loss

        return total_loss, [ent_loss, rel_loss, pol_loss], (ent_logit, rel_logit, pol_logit)