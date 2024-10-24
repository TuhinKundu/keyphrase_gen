from bert_score import BERTScorer
import json
import numpy as np
global scorer
import wordninja

from utilities.post_process_utils import stem_text


def json_load_one2seq(model, dataset):
    with open('graph_outputs/' + model + dataset + '_all_output.json', 'r') as f:
        dic = json.load(f)

    scores = dic['scores']
    predictions = dic['predictions']
    # entropies = dic['entropies']
    context_lines = dic['context_lines']

    return scores, predictions, context_lines

def clean_one2set(predictions, scores):
    new_preds, new_scores = [], []
    sep_token = '<sep>'
    useless_toks = ['<eos>', '<null>', '<digit>']
    flag=0
    for num, preds in enumerate(predictions):
        clean_preds = []
        clean_scores = []

        #print(preds)

        for i, tok in enumerate(preds):
            #print(i, tok)
            #print(predictions[i - 1])
            if tok in useless_toks:
                if flag==1:
                    flag=0
                    clean_preds.append(sep_token)
                    clean_scores.append(1.0)
                continue
            else:
                if i>0 and i<len(preds)-1:
                    if preds[i-1] not in useless_toks or preds[i+1] not in useless_toks:
                        clean_preds.append(tok)
                        clean_scores.append(scores[num][i])
                        flag=1

        new_preds.append(clean_preds)
        new_scores.append(clean_scores)

    return new_preds, new_scores

def json_load_one2set(model, dataset):
    with open('graph_outputs/' + model + dataset + '_all_output.json', 'r') as f:
        dic = json.load(f)

    scores = dic['scores']
    predictions = dic['predictions']
    context_lines = dic['context_lines']

    predictions, scores = clean_one2set(predictions, scores)
    return scores, predictions, context_lines

def load_exhird_preds(dataset, model='exhird_h_', seed=1):
    with open('data_dump/seed'+str(seed)+'/'+model+dataset+'_all_output.json', 'r') as f:
        dic= json.load(f)

    scores = dic['scores']
    predictions = dic['predictions']
    entropies = dic['entropies']

    return scores, predictions, entropies

def load_bart_preds(dataset):
    with open('graph_outputs/bart_'+dataset+'_all_output.json') as f:
        dic = json.load(f)

    kp_predictions=dic['kp_predictions']
    probabilities = dic['probabilities']
    predicted_tokens = dic['token_predictions']
    src = dic['src']
    targets = dic['targets']

    return src, kp_predictions, probabilities, predicted_tokens, targets

def load_bart_tokens(dataset, model='bart_'):
    with open('data_dump/'+model+'base/test_'+dataset+'_probs.json') as f:
        dic=json.load(f)



    probs = dic['probabilities']
    tokens = dic['token_predictions']
    #print(probs[0])
    #print(tokens[0])
    src = dic['src']
    target = dic['targets']


    return probs, tokens, src, target


import re
from collections import Counter

def word_prob(word): return dictionary[word] / total
def words(text): return re.findall('[a-z]+', text.lower())
#dictionary = Counter(words(open('big.txt').read()))
#max_word_length = max(map(len, dictionary))
#total = float(sum(dictionary.values()))

def viterbi_segment(text):
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]:i])
        i = lasts[i]
    words.reverse()
    return words, probs[-1]

def process_kps(kp_collect, model = 'bart_'):
    if model == 'bart_':
        return ''.join(kp_collect)
    else:
        kp = ''
        for i, token in enumerate(kp_collect):
            if token=='':
                kp+=' '
            elif len(token)<=3:
                kp+=token



def calculate_bart_ppl(scores, predictions, context_lines, model='bart_'):
    present_ppl, absent_ppl = [], []
    eos_token = '</s>'
    sep_token = ";"
    token_boundary =''
    if model != 'bart_':
        token_boundary =' '


    for i, pred in enumerate(predictions):
        stemmed_context = stem_text(context_lines[i])
        kp_collect = []
        prob_collect = []

        '''
        print(model, eos_token)
        print(pred)
        print(stemmed_context)
        print(scores[i])
        print(len(present_ppl))
        print(len(absent_ppl))
        print()
        '''
        kp_set = []
        present_kps = []
        absent_kps = []
        for j, token in enumerate(pred):
            if token.strip() in [eos_token, sep_token]:
                if len(kp_collect) > 0:
                    #print(prob_collect)
                    #print(kp_collect)
                    processed_kp = wordninja.split(''.join(kp_collect).replace(' ', ''))
                    #print(wordninja.split(''.join(kp_collect)))#.replace(' ', ''))
                    #print(processed_kp)
                    stemmed_kp = stem_text(' '.join(processed_kp))
                    #print(stemmed_kp)
                    if stemmed_kp not in kp_set:
                        kp_set.append(stemmed_kp)
                        ppl = np.prod(prob_collect) ** (-1 / float(len(prob_collect)))
                        #print(ppl)
                        #print()

                        #print(stemmed_kp)
                        if stemmed_kp in stemmed_context:
                            present_kps.append(stemmed_kp)
                            present_ppl.append(ppl)
                        else:
                            # print(stemmed_kp)
                            # print(stemmed_context)
                            # print(pred)
                            # print()
                            absent_kps.append(stemmed_kp)
                            absent_ppl.append(ppl)

                    kp_collect, prob_collect = [], []
                    if token.strip() == eos_token:
                        break
            else:
                kp_collect.append(token)
                prob_collect.append(scores[i][j])

        if len(kp_collect) > 0:
            processed_kp = wordninja.split(''.join(kp_collect).replace(' ', ''))
            stemmed_kp = stem_text(' '.join(processed_kp))
            ppl = np.prod(prob_collect) ** (-1 / float(len(prob_collect)))
            if stemmed_kp not in kp_set:
                if stemmed_kp in stemmed_context:
                    present_kps.append(stemmed_kp)
                    present_ppl.append(ppl)
                else:
                    absent_kps.append(stemmed_kp)
                    absent_ppl.append(ppl)

        #print("present kps: ", present_kps)
        #print("absent kps: ", absent_kps)
    #print(present_ppl)
    return present_ppl, absent_ppl

def calculate_bart_ppl_(scores, predictions, context_lines, model='bart_'):
    present_ppl, absent_ppl = [], []
    if model=='bart_':
        eos_token = ' ;'
    else:
        eos_token = ';'


    for i, pred in enumerate(predictions):
        stemmed_context = stem_text(context_lines[i])
        kp_collect = []
        prob_collect = []
        '''
        print(model, eos_token)
        print(pred)
        print(stemmed_context)
        print(scores[i])
        print(len(present_ppl))
        print(len(absent_ppl))
        print()
        '''
        kp_set = []
        for j, token in enumerate(pred):
            #print(token)
            if token == eos_token:
                if len(kp_collect) > 0:
                    #print(prob_collect)
                    #print(kp_collect)
                    stemmed_kp = stem_text(' '.join(kp_collect))
                    #print(stemmed_kp)
                    if stemmed_kp not in kp_set:
                        kp_set.append(stemmed_kp)
                        ppl = np.prod(prob_collect) ** (-1 / float(len(prob_collect)))
                    #print(ppl)
                    #print()

                    #print(stemmed_kp)
                        if stemmed_kp in stemmed_context:

                            present_ppl.append(ppl)
                        else:
                            # print(stemmed_kp)
                            # print(stemmed_context)
                            # print(pred)
                            # print()
                            absent_ppl.append(ppl)

                    kp_collect, prob_collect = [], []
            else:
                if len(pred) != len(scores[i]):
                    print(kp_collect, len(pred))
                    break
                kp_collect.append(token.strip())
                prob_collect.append(scores[i][j])
        if kp_collect[-1] == '</s>':
            kp_collect = kp_collect[:-1]
        if len(kp_collect) > 0:

            stemmed_kp = stem_text(' '.join(kp_collect))
            ppl = np.prod(prob_collect) ** (-1 / float(len(prob_collect)))
            if stemmed_kp not in kp_set:
                if stemmed_kp in stemmed_context:
                    present_ppl.append(ppl)
                else:
                    absent_ppl.append(ppl)
        print(model, eos_token)
        print(pred)
        print(stemmed_context)
        print(scores[i])
        print(len(present_ppl))
        print(len(absent_ppl))
        print()
    #print(present_ppl)

    return present_ppl, absent_ppl


def load_t5_preds(dataset, probab = False, seed=1):
    with open('data_dump/seed'+str(seed)+'/' + dataset + '_data.json', 'r') as f:
        dic = json.load(f)

    ppl = dic['perplexities']

    kp_predictions = dic['kp_predictions']
    probabilities = dic['probabilities']
    predicted_tokens = dic['token_predictions']

    context_lines = dic['src']


    for i, pred in enumerate(kp_predictions):

        j=0
        while j< len(pred):
            if pred[j] == '':
                pred.remove('')
            else:
                j+=1
        if len(pred) != len(ppl[i]):
            ppl[i] = ppl[i][:len(pred)]
    if probab == True:
        return ppl, kp_predictions, context_lines, probabilities, predicted_tokens
    else:
        return ppl, kp_predictions, context_lines



def format_exhird_predictions(predictions, scores):
    formatted_predictions = []
    formatted_ppl = []
    for a, kp_preds in enumerate(predictions):
        kp = ''
        formatted = []
        kp_ppl = []
        kp_scores = []
        for i, pred in enumerate(kp_preds):
            if pred == '<p_start>' or pred == '<a_start>':
                continue
            elif pred == ';':

                if len(kp)>0:
                    ppl = np.prod(kp_scores) ** (-1 / float(len(kp_scores)))
                    formatted.append(kp)
                    kp_ppl.append(ppl)
                kp = ''
                kp_scores = []
            else:
                kp_scores.append(scores[a][i])
                if kp =='':
                    kp+= pred
                else:
                    kp += ' ' + pred

        formatted_predictions.append(formatted)

        assert len(formatted)==len(kp_ppl)
        formatted_ppl.append(kp_ppl)
    return formatted_predictions, formatted_ppl



def get_exhird_targets(dataset):
    path = 'data/test_datasets/processed_'+dataset+'_testing_keyphrases.txt'

    #path = 'data/train_valid_dataset/processed_raw_data/processed_kp20k_training_keyphrases_filtered_PbfA_ordered_addBiSTokens_addSemicolon_RmStemDups_RmKeysAllUnk.txt'
    target_file = open(path, encoding='utf-8')
    target_lines = target_file.readlines()
    targets = []
    for i, preds in enumerate(target_lines):
        preds= preds.split(';')
        for j, pred in enumerate(preds):
            preds[j] = pred.strip()
        targets.append(preds)
    return targets



def get_exhird_context(dataset):
    path = 'data/test_datasets/processed_' + dataset + '_testing_context.txt'
    #path = 'data/train_valid_dataset/processed_raw_data/processed_kp20k_training_context_filtered_RmKeysAllUnk.txt'
    target_file = open(path, encoding='utf-8')
    target_lines = target_file.readlines()
    targets = []
    for i, preds in enumerate(target_lines):
        preds = preds.split(';')
        for j, pred in enumerate(preds):
            preds[j] = pred.strip()
        targets.append(preds)
    return targets


def json_output(results,filename):
    with open('graph_outputs/'+filename+'.json','w') as f:
        json.dump(results,f)


def initialize_bert_score_weights(lang, model_type):
    global scorer
    scorer = BERTScorer(lang=lang, rescale_with_baseline=True, model_type=model_type)

def return_bert_score(pred, target_kps):
    return scorer.score([pred for i in range(len(target_kps))], target_kps)

def FMFS_old(M, pred_len, gold_len):
    """
    Parameters
    ----------
    M = numpy matrix of shape pred_len x gold_len M[i,j] = bertscore(pred_phrase_i, gold_phrase_j)
    pred_len = scalar (no. of prediction phrases (phrases not words))
    gold_len = scalar (no. of gold phrases (phrases not words))

    Returns
    -------
    scalar score in [0,1] (I think)
    """
    alpha = 0.7
    beta = 0.7
    threshold_FMES = 0.4
    threshold_FMFS = 0.4
    epsilon = 1e-10

    pred2gold_scores = np.max(M, axis=1)
    pred2gold_scores = np.where(pred2gold_scores >= threshold_FMFS, pred2gold_scores, 0)

    denominator = pred_len + alpha * max(0, gold_len - pred_len)
    # print(denominator)

    pred2gold_score = np.sum(pred2gold_scores) / denominator

    gold2pred_scores = np.max(M, axis=0)
    gold2pred_scores = np.where(gold2pred_scores >= threshold_FMFS, gold2pred_scores, 0)

    denominator = gold_len + beta * max(0, pred_len - gold_len)
    gold2pred_score = np.sum(gold2pred_scores) / denominator

    score = (gold2pred_score + pred2gold_score) / 2

    return score


def FMFS(M, pred_len, gold_len):
    """
    Parameters
    ----------
    M = numpy matrix of shape pred_len x gold_len M[i,j] = bertscore(pred_phrase_i, gold_phrase_j)
    pred_len = scalar (no. of prediction phrases (phrases not words))
    gold_len = scalar (no. of gold phrases (phrases not words))

    Returns
    -------
    scalar score in [0,1] (I think)
    """
    alpha = 0.7
    beta = 0.7
    threshold_FMES = 0.4
    threshold_FMFS = 0.4
    epsilon = 1e-10

    pred2gold_scores = np.max(M, axis=1)
    pred2gold_scores = np.where(pred2gold_scores >= threshold_FMFS, pred2gold_scores, 0)
    denominator = pred_len

    pred2gold_score = np.sum(pred2gold_scores) / denominator if denominator > 0 else 0

    gold2pred_scores = np.max(M, axis=0)
    gold2pred_scores = np.where(gold2pred_scores >= threshold_FMFS, gold2pred_scores, 0)

    denominator = gold_len
    gold2pred_score = np.sum(gold2pred_scores) / denominator if denominator > 0 else 0

    denominator = gold2pred_score + pred2gold_score
    score = 2 * (gold2pred_score * pred2gold_score) / denominator if denominator > 0 else 0

    return score



def make_consistent(kp1, kp2):
    kp1_split = stem_text(kp1).split()
    kp2_split = stem_text(kp2).split()

    if len(kp2_split) == len(kp1_split):
        return kp1_split, kp2_split
    elif len(kp1_split) > len(kp2_split):
        for i in range(len(kp1_split) - len(kp2_split)):
            kp2_split.append("<dummy>")
    else:
        for i in range(len(kp2_split) - len(kp1_split)):
            kp1_split.append("dummy")
    return kp1_split, kp2_split

def generate_matrix(pred_kps, target_kps, func='bertscore'):
    matrix = []
    if func == 'bertscore':

        for i, pred in enumerate(pred_kps):
            P, R, F = return_bert_score(pred, target_kps)

            if target_kps == ['<dummy>']:
                matrix.append([0.0])
            else:
                matrix.append(F.tolist())
    else:
        for i, pred in enumerate(pred_kps):
            scores = []
            for j, target in enumerate(target_kps):
                hyp, ref = make_consistent(pred, target)
                if target =='<dummy>':
                    scores.append(0.0)
                else:
                    #print(hyp, ref)
                    #score1 = 1 - pyter.ter(hyp, ref)
                    #score2 = 1 - pyter.ter(ref, hyp)
                    scores.append(1 - pyter.ter(hyp, ref))
            matrix.append(scores)

    if matrix == []:
        matrix = [[0]]
    return matrix


def read_one2seq_targets(dataset):

    target_file_location = "data/one2set_testsets/"+dataset.lower()+"/test_trg.txt"
    targets = []
    for i, target_line in enumerate(open(target_file_location)):

        targets.append(target_line[:-1].split(';'))

    return targets


def preprocess_one2seq_predictions_to_kps(predictions):
    processed_kps = []
    for i, pred in enumerate(predictions):
        kp_collect = []
        kps = []
        for j, token in enumerate(pred):
            if token == '<sep>':
                if len(kp_collect) > 0:
                    kps.append(' '.join(kp_collect))
                    kp_collect = []
            else:
                kp_collect.append(token)
        if len(kp_collect) > 0:
            kps.append(' '.join(kp_collect))
        processed_kps.append(kps)
    return processed_kps
