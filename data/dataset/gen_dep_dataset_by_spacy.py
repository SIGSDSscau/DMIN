import json
from itertools import accumulate
import numpy as np
from tqdm import tqdm
import spacy
import re
from collections import defaultdict
import copy

def align_index_with_list(tokenizer, tokens):
    pieces2word = []
    word_num = 0
    all_pieces = []

    tokens = [tokenizer.tokenize(w) for w in tokens]
    cur_line = []
    for token in tokens:
        for piece in token:
            pieces2word.append(word_num)
        word_num += 1
        cur_line += token
    all_pieces.append(cur_line)
    
    return all_pieces, pieces2word

def find_utterance_index(replies, sentence_lengths):
    utterance_collections = [i for i, w in enumerate(replies) if w == 0]
    zero_index = utterance_collections[0]
    for i in range(len(replies)):
        if i < zero_index: continue
        if replies[i] == 0:
            zero_index = i
        replies[i] = (i - zero_index)
    sentence_index = [w + 1 for w in replies]
    utterance_index = [[w] * z for w, z in zip(sentence_index, sentence_lengths)]
    utterance_index = [w for line in utterance_index for w in line]
    token_index = [list(range(sentence_lengths[0]))]
    lens = len(token_index[0])
    for i, w in enumerate(sentence_lengths):
        if i == 0: continue
        if sentence_index[i] == 1:
            distance = lens
        token_index += [list(range(distance, distance + w))]
        distance += w
    token_index = [w for line in token_index for w in line]
    utterance_collections = np.split(sentence_index, utterance_collections)
    thread_nums = list(map(len, utterance_collections))
    thread_ranges = [0] + list(accumulate(thread_nums))
    thread_lengths = [sum(sentence_lengths[thread_ranges[i]:thread_ranges[i+1]]) for i in range(len(thread_ranges)-1)]
    sent_idx2reply_idx = defaultdict(int)
    for sent_idx, reply in enumerate(replies):
        if reply == -1:
            sent_idx2reply_idx[sent_idx] = 0
        elif reply == 0:
            sent_idx2reply_idx[sent_idx] = 0
        else:
            sent_idx2reply_idx[sent_idx] = last_reply_idx
        last_reply_idx = sent_idx

    return utterance_index, token_index, thread_lengths, thread_nums, sent_idx2reply_idx

def get_word2firstpiece(tokenizer, doc):
    word2firstpiece = {}
    total_pieces_num = 0
    cur_word_idx = 0
    for sen_idx, sub_sen in enumerate(doc.sentences):
        for dp in sub_sen.dependencies:
            pieces = tokenizer.tokenize(dp[2].text)
            word2firstpiece[cur_word_idx] = total_pieces_num
            total_pieces_num += len(pieces)
            cur_word_idx += 1
    return word2firstpiece

from transformers import AutoTokenizer


def align_index_by_dep2piece(labels, ori_token2dep_piece, merge_pieces, unknown_token, tokenizer):
    
    if len(labels) == 0: return []

    new_labels = []
    if len(labels[0]) == 3:
        for i, (s, e, l) in enumerate(labels):
            ns = ori_token2dep_piece[s][0]
            ne = ori_token2dep_piece[e - 1][-1] + 1
            l = re.sub('[' + re.escape(unknown_token) + ']', 'UNK', l)
            new_labels.append([ns, ne, l])
            l = "".join(tokenizer.tokenize(l)).replace("##", "").replace("Ġ", "")
            assert l.replace(" ", "").lower() == "".join(merge_pieces[ns:ne]).replace("##", "").lower()
        return new_labels
    elif len(labels[0]) == 4:
        for i, (s, e, l, p) in enumerate(labels):
            ns = ori_token2dep_piece[s][0]
            ne = ori_token2dep_piece[e - 1][-1] + 1
            l = re.sub('[' + re.escape(unknown_token) + ']', 'UNK', l)
            new_labels.append([ns, ne, l, p])
            l = "".join(tokenizer.tokenize(l)).replace("##", "").replace("Ġ", "")
            assert l.replace(" ", "").lower() == "".join(merge_pieces[ns:ne]).replace("##", "").lower()
        return new_labels

    elif len(labels[0]) == 10:
        for i, (t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t) in enumerate(labels):
            nts, nas, nos = [ori_token2dep_piece[w][0] if w != -1 else -1 for w in [t_s, a_s, o_s]]
            nte, nae, noe = [ori_token2dep_piece[w - 1][-1] + 1 if w != -1 else -1 for w in [t_e, a_e, o_e]]
            o_t = re.sub('[' + re.escape(unknown_token) + ']', 'UNK', o_t)
            t_t = re.sub('[' + re.escape(unknown_token) + ']', 'UNK', t_t)
            a_t = re.sub('[' + re.escape(unknown_token) + ']', 'UNK', a_t)
            new_labels.append((nts, nte, nas, nae, nos, noe, polarity, t_t, a_t, o_t))

            t_t, a_t, o_t = "".join(tokenizer.tokenize(t_t)).replace("##", ""), "".join(tokenizer.tokenize(a_t)).replace("##", ""), "".join(tokenizer.tokenize(o_t)).replace("##", "")
            # roberta tokenizer will replace the ' ' with 'Ġ'
            t_t, a_t, o_t = t_t.replace("Ġ", ""), a_t.replace("Ġ", ""), o_t.replace("Ġ", "")

            assert t_t.replace(" ", "").lower() == "".join(merge_pieces[nts:nte]).replace("##", "").lower(), f"{t_t.replace(' ', '').lower()} != {''.join(merge_pieces[nts:nte]).replace('##', '').lower()}"
            assert a_t.replace(" ", "").lower() == "".join(merge_pieces[nas:nae]).replace("##", "").lower(), f"{a_t.replace(' ', '').lower()} != {''.join(merge_pieces[nas:nae]).replace('##', '').lower()}"
            assert o_t.replace(" ", "").lower() == "".join(merge_pieces[nos:noe]).replace("##", "").lower(), f"{o_t.replace(' ', '').lower()} != {''.join(merge_pieces[nos:noe]).replace('##', '').lower()}"
        return new_labels
    else:
        raise ValueError("labels length error")

def get_special_token(dataset, tokenizer):
    tmp_set = set()
    for dialog in tqdm(dataset):
        sentences  = dialog['sentences']
        for sent in sentences:
            # put the characters that are not in the vocab into the special_tokens
            for char in sent:
                tmp_set.add(char)
    special_tokens = set()
    for char in tmp_set:
        if char not in tokenizer.vocab:
            special_tokens.add(char)
            
    return special_tokens



def gen_dep_dataset(dataset_path, bert_path=None):
    if 'en' in dataset_path:
        lang = 'en'
        cls_ = '<s>'
        sep_ = '</s>'
        # spacy_model = 'en_core_web_md'
        spacy_model = 'en_core_web_trf'
        unknown_token = '·×欢水赴度合—鱼共巫╮揖山≥▽╭≈☞🍔—🐛🙉🙄🔨🏆🆔👌👀😎🥺冖🌚🙈😭🍎😅💩尛硌糇💰🐴🙊💯⭐🐶🐟🙏😄🏻📶🐮🍺❌🤔🐍🐸🙃🤣🏆😂🌚'
        # skip_tokens = [cls_, sep_, "+", '[', ']', '][', '.', '..', '...', '....', '.....', '......', '......................', '.......................']
    elif 'zh' in dataset_path:
        lang = 'zh'
        cls_ = '[CLS]'
        sep_ = '[SEP]'
        # spacy_model = 'zh_core_web_md'
        spacy_model = 'zh_core_web_trf'
        unknown_token = '🍔—🐛🙉🙄🔨🏆🆔👌👀🥺冖🌚🙈😭🍎😅💩尛硌糇💰😎🐴🙊💯⭐🐶🐟🙏😄🏻📶🐮🍺❌🤔🐍🐸🙃🤣🏆😂🌚崼湉戆犇恾囖秫–揾铼…獬瘆—魉碜莜豸姮诹嚯’”慜“炘媺魑郓'
        # skip_tokens = set([cls_, "+", '.', '..', '...', '....', '.....', '......', '......................', '.......................',
            #  '52', '30', '5230', '28', '89', '2889', '84', '75', '8475', '98', '20', '9820', '8', '70', '870', '88', '90', '8890', '33', '99', '3399'])

    

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    # load the spacy model
    nlp = spacy.load(spacy_model)
    # load the dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    new_dataset = []
    num_cross_sentence = 0
    for dialog in tqdm(dataset, desc=f'{dataset_path}'):
        sentences, speakers, replies, = [dialog[w] for w in ['sentences', 'speakers', 'replies']]
        sentence_length = list(map(lambda x : len(x) + 2, sentences))
        utterance_index, token_index, thread_length, thread_nums, sent_idx2reply_idx = find_utterance_index(replies, sentence_length)
        root_piece_idxes, heads, deprels, poss, dep_tokens, pieces, dep_token2pieces = [], [], [], [], [], [], []
        # 0 replace unknown token with 'UNK' in the sentence
        sentences = [re.sub('[' + re.escape(unknown_token) + ']', 'UNK', sentence) for sentence in sentences]
        # 1 depenency parse, tokenize as pieces
        for s_idx, sentence in enumerate(sentences): # each sentence
            
            if 'zh' in dataset_path:
                sentence = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', sentence)
            doc = nlp(sentence)
            root_piece_idx, head, deprel, pos, dep_token, piece, dep_token2piece = [], [], [], [], [], [], defaultdict(list)
            
            cur_piece = 0
            for i, token in enumerate(doc):
                for pie_idx, pie in enumerate(tokenizer.tokenize(token.text)) :
                    if token.dep_ == 'ROOT' and pie_idx == 0:
                        root_piece_idx.append(cur_piece)
                    if pie_idx == 0:
                        first_idx_of_token = cur_piece
                        h = token.head.i       # head.i is the index of the head token
                        dep = token.dep_
                    else:
                        h = i   # piece head is the token head
                        dep = 'piece'
                    head.append(h)
                    deprel.append(dep)
                    pos.append(token.pos_)
                    dep_token.append(token.text)
                    piece.append(pie)
                    dep_token2piece[i].append(cur_piece)
                    cur_piece += 1
            head = [dep_token2piece[h][0] for h in head]
            root_piece_idxes.append(root_piece_idx[0])  # first piece of first root token of each sentence
            heads.append(head)
            deprels.append(deprel)
            poss.append(pos)
            dep_tokens.append(dep_token)
            pieces.append(piece)
            dep_token2pieces.append(dep_token2piece)
        # 2 spacy tokenize -> bert tokenize -> piece    align
        ori_tokens = [sent.split(' ') for sent in sentences]
        
        assert len(ori_tokens) == len(pieces)
        ori_token2dep_piece = defaultdict(list)
        cur_dep_pieces_len = 0
        cur_ori_tokens_len= 0
        merge_pieces = []
        for ori_token, dep_piece, dep_token2piece in zip(ori_tokens, pieces, dep_token2pieces):
            merge_pieces.extend(dep_piece)
            bia_j = 0
            for i,j in zip(range(len(ori_token)), range(len(dep_piece))): 
                assert len(ori_token[i]) >= len(dep_piece[j+bia_j]) or '[UNK]' in dep_piece[j+bia_j] or '[UNK]' in ori_token[i], (ori_token[i], dep_piece[j+bia_j])
                ori_token2dep_piece[i+cur_ori_tokens_len].append(j+bia_j+cur_dep_pieces_len)
                dep_term = dep_piece[j+bia_j].strip()
                # align the token after bert tokenization
                ori_token_after_tokenize = ''.join(tokenizer.tokenize(ori_token[i])).replace("##", "").replace("Ġ", "")
                # while ori_token[i].lower() != dep_term.lower():
                while ori_token_after_tokenize.lower() != dep_term.lower():
                    bia_j+=1
                    if 'en' in dataset_path:
                        dep_term += dep_piece[j+bia_j]
                    else :
                        dep_term += dep_piece[j+bia_j].replace("##", "").strip()
                    ori_token2dep_piece[i+cur_ori_tokens_len].append(j+bia_j+cur_dep_pieces_len)
            cur_ori_tokens_len += len(ori_token) 
            cur_dep_pieces_len += len(dep_piece) 
            assert (i == len(ori_token)-1 and len(dep_piece)-1 == j+bia_j), f"index diff: {i} {j+bia_j}"
        
        dep_piece2ori_token = [k for k,v in ori_token2dep_piece.items() for _ in range(len(v))]
        assert len(ori_token2dep_piece) == sum([len(s) for s in ori_tokens]), f"ori_token2dep_piece: {len(ori_token2dep_piece)} ori_tokens: {sum([len(s) for s in ori_tokens])}"
        assert ori_token2dep_piece[len(ori_token2dep_piece)-1][-1] == len(merge_pieces)-1, f"ori_token2dep_piece: {ori_token2dep_piece[len(ori_token2dep_piece)-1][-1]} pieces: {len(merge_pieces)-1}"
        
        # 3 align the index of triplets, targets, aspects, opinions
        triplets, targets, aspects, opinions = [dialog[w] for w in ['triplets', 'targets', 'aspects', 'opinions']]
        triplets = align_index_by_dep2piece(triplets, ori_token2dep_piece, merge_pieces,unknown_token, tokenizer)
        targets = align_index_by_dep2piece(targets, ori_token2dep_piece, merge_pieces,unknown_token, tokenizer)
        aspects = align_index_by_dep2piece(aspects, ori_token2dep_piece, merge_pieces,unknown_token, tokenizer)
        opinions = align_index_by_dep2piece(opinions, ori_token2dep_piece, merge_pieces,unknown_token, tokenizer)
        # 4 get the dialog split by thread
        heads_with_cls_sep = [[0]+[h+1 for h in hs ]+[len(hs)+1] for hs in heads]
        pieces_with_cls_sep = [[cls_]+pie+[sep_] for pie in pieces]
        poss_with_cls_sep = [['SENT_BEGIN']+ps+['SENT_END'] for ps in poss]
        deprels_with_cls_sep = [['SENT_BEGIN']+ds+['SENT_END'] for ds in deprels]
        ori_tokens_with_cls_sep = [[cls_]+ori_token+[sep_] for ori_token in ori_tokens]
        thread_pieces, thread_heads, thread_poss, thread_deprels, thread_ori_tokens, thread_dep_token2pieces = [], [], [], [], [], []
        thread_range = list(accumulate(thread_nums))
        for i, idx in enumerate(thread_range): # copy the root for each thread
            if i == 0 : 
                continue
            # threads.append(sentences[0]+" "+" ".join(sentences[thread_range[i-1]:thread_range[i]]))
            thread_pieces.append(copy.deepcopy(pieces_with_cls_sep[0]))
            thread_heads.append(copy.deepcopy(heads_with_cls_sep[0]))
            thread_poss.append(copy.deepcopy(poss_with_cls_sep[0]))
            thread_deprels.append(copy.deepcopy(deprels_with_cls_sep[0]))
            thread_ori_tokens.append(copy.deepcopy(ori_tokens_with_cls_sep[0]))
            # thread_dep_token2pieces.append(dep_token2pieces[0])
            root_piece_idx = []
            
            for j in range(thread_range[i-1], thread_range[i]):
                j_th_head = [h + len(thread_pieces[i-1]) for h in heads_with_cls_sep[j]]
                thread_heads[i-1].extend(j_th_head)
                thread_pieces[i-1].extend(pieces_with_cls_sep[j])
                thread_poss[i-1].extend(poss_with_cls_sep[j])
                thread_deprels[i-1].extend(deprels_with_cls_sep[j])
                thread_ori_tokens[i-1].extend(ori_tokens_with_cls_sep[j])
            assert len(thread_heads[i-1]) == len(thread_pieces[i-1]) and len(thread_heads[i-1]) == len(thread_poss[i-1]) and len(thread_heads[i-1]) == len(thread_deprels[i-1])
        
        
        dialog['piece_dep'] = {
            'pieces': pieces,                   # without cls and sep
            'heads': heads_with_cls_sep,        # with cls and sep
            'deprels': deprels_with_cls_sep,
            'poss': poss_with_cls_sep,       
            'thread_pieces': thread_pieces,
            'thread_heads': thread_heads,
            'thread_deprels': thread_deprels,
            'thread_poss': thread_poss,
            # 'ori_tokens': ori_tokens,
            'dep_piece2ori_token': dep_piece2ori_token,     
            'triplets': triplets,
            'targets': targets,
            'aspects': aspects,
            'opinions': opinions,
        }
        dialog['local_dependency'] = ''
        dialog['dependency'] = ''
    # write the new dataset to json file
    with open(dataset_path.replace('.json', '_dependent_trf.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    

errors = {}
import concurrent.futures

def main():
    bert_path_zh = '/home/shaw/hfl/chinese-roberta-wwm-ext'
    bert_path_en = '/home/shaw/hfl/roberta-large'
    tasks = [('jsons_zh/train.json', bert_path_zh),
             ('jsons_zh/valid.json', bert_path_zh),
             ('jsons_zh/test.json', bert_path_zh),
             ('jsons_en/train.json', bert_path_en),
             ('jsons_en/valid.json', bert_path_en),
             ('jsons_en/test.json', bert_path_en)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(gen_dep_dataset, task[0], task[1]) for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Generated an exception: {e}")

if __name__ == "__main__":
    main()
