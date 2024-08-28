import json

import numpy as np
#from generate_graphs import *

from generate_graphs_transformers import *
#from utilities.post_process_utils import *
#from fuzzy_matching import *
from utilities.utils import *
from tqdm import tqdm
from transformers import T5Tokenizer, BartTokenizer



def calibrate_one2set(dataset, num_buckets=20, model2 = 'one2set_'):

    prob_scores, predictions, context_lines = json_load_one2set(model2, dataset.lower())

    targets = read_one2seq_targets(dataset)
    #present_ppl, absent_ppl = get_one2seq_ppl(raw_predictions, prob_scores, context_lines)
    #predictions = preprocess_one2seq_predictions_to_kps(predictions)

    present_samples = 0
    present_bucket_samples = np.array([0 for _ in range(num_buckets)])
    present_bucket_confidence = np.array([0.0 for _ in range(num_buckets)])
    present_bucket_acc = np.array([0 for _ in range(num_buckets)])

    absent_samples = 0
    absent_bucket_samples = np.array([0 for _ in range(num_buckets)])
    absent_bucket_confidence = np.array([0.0 for _ in range(num_buckets)])
    absent_bucket_acc = np.array([0 for _ in range(num_buckets)])

    separator_token = '<sep>'

    print('one2set predictions processing...')

    with tqdm(total=len(targets), leave=True) as pbar:
        for i, context in enumerate(context_lines):

            present_preds, absent_preds = segregate_kps_for_one2set(predictions[i], context)
            present_targets, absent_targets = segregate_kps(targets[i], context)

            present_preds = remove_duplicates(present_preds)
            absent_preds = remove_duplicates(absent_preds)
            present_samples += len(present_preds)
            absent_samples += len(absent_preds)

            kpp_inverse = 1
            kp_collect = []
            set_ = set()
            for j, token in enumerate(predictions[i]):

                if token != separator_token:
                    kpp_inverse*=prob_scores[i][j]
                    kp_collect.append(token)


                else:
                    if len(kp_collect)==0:
                        continue
                    bucket_pos = get_bucket(kpp_inverse ** (1 / len(kp_collect)), num_buckets)
                    keyphrase= ' '.join(kp_collect)
                    stemmed_kp_pred = stem_text(keyphrase)#stem_text(keyphrase)
                    #print(stemmed_kp_pred, bucket_pos)
                    if stemmed_kp_pred not in set_:
                        set_.add(stemmed_kp_pred)
                        if stemmed_kp_pred in present_preds:

                            present_bucket_confidence[bucket_pos] += kpp_inverse ** (1 / len(kp_collect))
                            present_bucket_samples[bucket_pos] += 1
                            if stemmed_kp_pred in present_targets:
                                present_bucket_acc[bucket_pos] += 1
                        else:
                            absent_bucket_confidence[bucket_pos] += kpp_inverse ** (1 / len(kp_collect))
                            absent_bucket_samples[bucket_pos] += 1
                            if stemmed_kp_pred in absent_targets:
                                absent_bucket_acc[bucket_pos] += 1

                    kpp_inverse=1
                    kp_collect=[]

            pbar.update(1)

    ece, mce, tce = calculate_error(present_samples + absent_samples,
                                    present_bucket_samples + absent_bucket_samples,
                                    (present_bucket_confidence + absent_bucket_confidence) / (
                                                present_bucket_samples + absent_bucket_samples),
                                    (present_bucket_acc + absent_bucket_acc) / (
                                                present_bucket_samples + absent_bucket_samples))
    print(ece)
    present_ece, present_mce, present_tce = calculate_error(present_samples, present_bucket_samples,
                                                            present_bucket_confidence / present_bucket_samples,
                                                            present_bucket_acc / present_bucket_samples)
    absent_ece, absent_mce, absent_tce = calculate_error(absent_samples, absent_bucket_samples,
                                                         absent_bucket_confidence / absent_bucket_samples,
                                                         absent_bucket_acc / absent_bucket_samples)
    print(f'present_ece: {present_ece}')
    print(f'absent_ece: {absent_ece}')

    print()
    return (present_bucket_acc + absent_bucket_acc) / (present_bucket_samples + absent_bucket_samples)


def calibrate_exhird(dataset, num_buckets=20):
    exhird_probs, raw_exhird_predictions, entropies = load_exhird_preds(dataset)
    exhird_context_lines = get_exhird_context(dataset)
    exhird_predictions, _ = format_exhird_predictions(raw_exhird_predictions, exhird_probs)
    exhird_targets = get_exhird_targets(dataset)


    present_samples = 0
    present_bucket_samples = np.array([0 for _ in range(num_buckets)])
    present_bucket_confidence = np.array([0.0 for _ in range(num_buckets)])
    present_bucket_acc = np.array([0 for _ in range(num_buckets)])


    absent_samples = 0
    absent_bucket_samples = np.array([0 for _ in range(num_buckets)])
    absent_bucket_confidence = np.array([0.0 for _ in range(num_buckets)])
    absent_bucket_acc = np.array([0 for _ in range(num_buckets)])

    print('Exhird predictions processing: ')
    with tqdm(total=len(exhird_targets), leave=True) as pbar:
        for i, exhird_context in enumerate(exhird_context_lines):
            present_preds, absent_preds = segregate_kps(exhird_predictions[i], exhird_context[0])
            present_targets, absent_targets = segregate_kps(exhird_targets[i], exhird_context[0])

            present_preds = remove_duplicates(present_preds)
            absent_preds = remove_duplicates(absent_preds)
            present_samples += len(present_preds)
            absent_samples += len(absent_preds)

            #present_matrix = generate_matrix(exhird_predictions[i], exhird_targets[i], func='ter')

            #num_samples += len(exhird_predictions[i])
            # partial_vector, wrong_pred = partial_match_vector(present_preds, present_targets, present_matrix)


            kpp_inverse = 1
            num_tokens = 0
            kp_type = ''
            collect_tokens= ''
            set_ = set()
            for j, token in enumerate(raw_exhird_predictions[i]):

                if '_start' in token:

                    kpp_inverse = 1
                    num_tokens = 0
                    collect_tokens = ''
                    if 'p_start' in token:
                        kp_type = 'present'
                    else:
                        kp_type = 'absent'
                    continue

                if token == ';':
                    if num_tokens > 0:
                        bucket_pos = get_bucket(kpp_inverse**(1/num_tokens), num_buckets)
                        stem_collect_tokens = stem_text(collect_tokens)
                        if kp_type == 'present':

                            present_bucket_confidence[bucket_pos] += kpp_inverse**(1/num_tokens)
                            present_bucket_samples[bucket_pos] += 1

                            if stem_collect_tokens in stem_text(' '.join(present_targets)) and stem_collect_tokens not in set_:
                                present_bucket_acc[bucket_pos] += 1
                                set_.add(stem_collect_tokens)
                        else:
                            absent_bucket_confidence[bucket_pos] += kpp_inverse**(1/num_tokens)
                            absent_bucket_samples[bucket_pos] += 1

                            if  stem_collect_tokens in stem_text(' '.join(absent_targets)) and stem_collect_tokens not in set_:
                                absent_bucket_acc[bucket_pos] += 1
                                set_.add(stem_collect_tokens)

                    continue

                # if num_kp > -1:
                kpp_inverse *= exhird_probs[i][j]
                num_tokens += 1
                if len(collect_tokens) == 0:
                    collect_tokens = token
                else:
                    collect_tokens += ' ' + token
                '''
                bucket_pos = get_bucket(exhird_probs[i][j], num_buckets)
                present_bucket_samples[bucket_pos] += 1
                present_bucket_confidence[bucket_pos] += exhird_probs[i][j]
                #max_value_pos = np.argmax(present_matrix[num_kp])
                #if present_matrix[num_kp][max_value_pos] > partial_threshold:

                if stem_text(token) in stem_text(' '.join(exhird_targets[i])):
                    #if exhird_probs[i][j] < 0.2:
                        #print(token, exhird_probs[i][j], exhird_targets[i])
                    present_bucket_acc[bucket_pos] += 1
                '''
            pbar.update(1)

    '''print(present_samples)
    print(absent_samples)
    print(present_bucket_samples)
    print(absent_bucket_samples)
    print(present_bucket_acc, np.sum(present_bucket_acc))
    print(absent_bucket_acc, np.sum(absent_bucket_acc))
    print((present_bucket_confidence + absent_bucket_confidence) / (present_bucket_samples + absent_bucket_samples))

    print((present_bucket_acc + absent_bucket_acc) / (present_bucket_samples + absent_bucket_samples))

    '''
    ece,mce,tce = calculate_error(present_samples+absent_samples, present_bucket_samples+absent_bucket_samples ,
                                  (present_bucket_confidence+absent_bucket_confidence) / (present_bucket_samples+absent_bucket_samples),
                                  (present_bucket_acc+absent_bucket_acc) / (present_bucket_samples+absent_bucket_samples))
    print('ece:',ece)
    present_ece, present_mce, present_tce = calculate_error(present_samples, present_bucket_samples,
                                                            present_bucket_confidence / present_bucket_samples,
                                                            present_bucket_acc / present_bucket_samples)
    absent_ece, absent_mce, absent_tce = calculate_error(absent_samples, absent_bucket_samples,
                                                         absent_bucket_confidence / absent_bucket_samples,
                                                         absent_bucket_acc / absent_bucket_samples)
    print(f'present_ece: {present_ece}')
    print(f'absent_ece: {absent_ece}')

    print()
    return (present_bucket_acc+absent_bucket_acc) / (present_bucket_samples+absent_bucket_samples)



def calibrate_t5(dataset, num_buckets=10, tokenizer='t5-tokenizer/'):
    ppl, t5_predictions, t5_context_lines, t5_probs, t5_tokens = load_t5_preds(dataset, probab=True)

    t5_targets = get_t5_targets(dataset)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer)
    sep_token = tokenizer.encode('<sep>', add_special_tokens=False)[0]

    present_samples = 0
    present_bucket_samples = np.array([0 for _ in range(num_buckets)])
    present_bucket_confidence = np.array([0.0 for _ in range(num_buckets)])
    present_bucket_acc = np.array([0 for _ in range(num_buckets)])

    absent_samples = 0
    absent_bucket_samples = np.array([0 for _ in range(num_buckets)])
    absent_bucket_confidence = np.array([0.0 for _ in range(num_buckets)])
    absent_bucket_acc = np.array([0 for _ in range(num_buckets)])
    present_correct, absent_correct = 0,0

    with tqdm(total=len(t5_context_lines), leave=True) as pbar:

        for i, preds in enumerate(t5_predictions):
            present_preds, absent_preds = segregate_kps(remove_duplicates(t5_predictions[i]), t5_context_lines[i])
            present_targets, absent_targets = segregate_kps(t5_targets[i], t5_context_lines[i])
            #print(present_preds)
            #print(absent_preds)
            #print()
            present_samples += len(present_preds)
            absent_samples += len(absent_preds)

            stem_present_targets = [stem_text(kp) for kp in present_targets]
            stem_absent_targets = [stem_text(kp) for kp in absent_targets]

            stem_present_preds = [stem_text(kp) for kp in present_preds]
            stem_absent_preds = [stem_text(kp) for kp in absent_preds]


            ''''print(stem_present_preds)
            print(stem_present_targets)
            print(stem_absent_preds)
            print(stem_absent_targets)
            print()'''

            for pred in present_preds:
                if stem_text(pred) in stem_present_targets:
                    present_correct+=1
            for pred in absent_preds:
                if stem_text(pred) in stem_absent_targets:
                    absent_correct +=1

            idx_tokens = t5_tokens[i][0]

            kpp_inverse = 1
            num_kp = 0

            collect_tokens = ''
            seen_preds = []
            for j, token_id in enumerate(idx_tokens):

                if token_id == tokenizer.pad_token_id:
                    continue
                #print(token_id, tokenizer.decode(token_id))
                if token_id == sep_token or token_id == tokenizer.eos_token_id:
                    if len(collect_tokens)>0:
                        #print(collect_tokens, num_tokens)
                        #print()
                        kp_pred = t5_predictions[i][num_kp]
                        num_kp += 1
                        num_words = len(kp_pred.strip().split())
                        bucket_pos = get_bucket((kpp_inverse ** (1 / num_words)), num_buckets)

                        #print(collect_tokens, kpp_inverse**(1/num_tokens))
                        if kp_pred not in seen_preds and 'dummy' not in kp_pred:
                            seen_preds.append(kp_pred)
                            if stem_text(kp_pred) in stem_text(t5_context_lines[i]):

                                present_bucket_confidence[bucket_pos] += (kpp_inverse ** (1 / num_words))
                                present_bucket_samples[bucket_pos] += 1

                                if stem_text(kp_pred) in stem_present_targets:
                                    present_bucket_acc[bucket_pos] += 1
                            else:
                                #print(kp_pred, collect_tokens,num_tokens, kpp_inverse**(1/num_tokens))

                                absent_bucket_confidence[bucket_pos] += (kpp_inverse ** (1 / num_words))
                                absent_bucket_samples[bucket_pos] += 1
                                if stem_text(kp_pred) in stem_absent_targets:
                                    absent_bucket_acc[bucket_pos] += 1

                        kpp_inverse = 1

                        collect_tokens = ''

                else:
                    #print(tokenizer.decode(idx_tokens[j]))
                    kpp_inverse *= t5_probs[i][j]
                    collect_tokens += tokenizer.decode(token_id, skip_special_tokens=False)

            pbar.update(1)
    '''
        print(present_samples)
        print(absent_samples)
        print(present_bucket_samples)
        print(absent_bucket_samples)
        print(present_correct, present_bucket_acc, np.sum(present_bucket_acc))
        print(absent_correct, absent_bucket_acc, np.sum(absent_bucket_acc))
        print((present_bucket_confidence + absent_bucket_confidence) / (present_bucket_samples + absent_bucket_samples))
    '''

    print((present_bucket_acc + absent_bucket_acc) / (present_bucket_samples + absent_bucket_samples))

    ece, mce, tce = calculate_error(present_samples + absent_samples, present_bucket_samples + absent_bucket_samples,
                                    (present_bucket_confidence + absent_bucket_confidence) / (
                                                present_bucket_samples + absent_bucket_samples),
                                    (present_bucket_acc + absent_bucket_acc) / (
                                                present_bucket_samples + absent_bucket_samples))
    print('ECE:',ece)
    present_ece, present_mce, present_tce = calculate_error(present_samples, present_bucket_samples,
                                                            present_bucket_confidence / present_bucket_samples,
                                                            present_bucket_acc / present_bucket_samples)
    absent_ece, absent_mce, absent_tce = calculate_error(absent_samples, absent_bucket_samples,
                                                         absent_bucket_confidence / absent_bucket_samples,
                                                         absent_bucket_acc / absent_bucket_samples)
    print(f'present_ece: {present_ece}')
    print(f'absent_ece: {absent_ece}')

    print()
    return (present_bucket_acc + absent_bucket_acc) / (present_bucket_samples + absent_bucket_samples)



def calibrate_bart(dataset, num_buckets=10, model = 'facebook/bart-base'):

    t5_context_lines, t5_predictions, t5_probs, t5_tokens, t5_targets = load_bart_preds(dataset)

    #t5_targets = get_t5_targets(dataset)
    tokenizer = BartTokenizer.from_pretrained(model)
    sep_token = '<s>'
    eos_token = ' ;'

    present_samples = 0
    present_bucket_samples = np.array([0 for _ in range(num_buckets)])
    present_bucket_confidence = np.array([0.0 for _ in range(num_buckets)])
    present_bucket_acc = np.array([0 for _ in range(num_buckets)])

    absent_samples = 0
    absent_bucket_samples = np.array([0 for _ in range(num_buckets)])
    absent_bucket_confidence = np.array([0.0 for _ in range(num_buckets)])
    absent_bucket_acc = np.array([0 for _ in range(num_buckets)])
    present_correct, absent_correct = 0,0

    with tqdm(total=len(t5_context_lines), leave=True) as pbar:

        for i, preds in enumerate(t5_predictions):

            present_preds, absent_preds = segregate_kps(remove_duplicates(t5_predictions[i]), t5_context_lines[i])
            present_targets, absent_targets = segregate_kps(t5_targets[i], t5_context_lines[i])
            #print(present_targets, absent_targets)
            present_samples += len(present_preds)
            absent_samples += len(absent_preds)

            stem_present_preds = [stem_text(kp) for kp in present_preds]
            stem_absent_preds = [stem_text(kp) for kp in absent_preds]

            stem_present_targets = [stem_text(kp) for kp in present_targets]
            stem_absent_targets = [stem_text(kp) for kp in absent_targets]

            '''print(stem_present_preds)
            print(stem_present_targets)
            print(stem_absent_preds)
            print(stem_absent_targets)
            print()'''

            for pred in present_preds:
                stem_pred = stem_text(pred)
                if stem_pred in stem_present_targets:
                    present_correct+=1
            for pred in absent_preds:
                stem_pred=stem_text(pred)
                if stem_pred in stem_absent_targets:
                    absent_correct +=1


            tokens = t5_tokens[i]



            kpp_inverse = 1
            num_kp = 0

            collect_tokens = ''
            seen_preds = []
            for j, token in enumerate(tokens):

                #print(token_id, tokenizer.decode(token_id))
                if token == sep_token or token == eos_token:
                    if len(collect_tokens)>0:
                        #print(collect_tokens, num_tokens)
                        #print()
                        kp_pred = t5_predictions[i][num_kp]
                        num_kp += 1
                        num_words = len(collect_tokens.strip().split())
                        bucket_pos = get_bucket((kpp_inverse ** (1 / num_words)), num_buckets)

                        #print(collect_tokens, kpp_inverse**(1/num_tokens))
                        if kp_pred not in seen_preds and 'dummy' not in kp_pred:
                            seen_preds.append(kp_pred)
                            if stem_text(kp_pred) in stem_text(t5_context_lines[i]):

                                present_bucket_confidence[bucket_pos] += (kpp_inverse ** (1 / num_words))
                                present_bucket_samples[bucket_pos] += 1

                                #print(kp_pred)
                                #print(stem_present_targets)
                                #print()
                                if stem_text(kp_pred) in stem_present_targets:
                                    present_bucket_acc[bucket_pos] += 1
                            else:
                                #print(kp_pred, collect_tokens,num_tokens, kpp_inverse**(1/num_tokens))

                                absent_bucket_confidence[bucket_pos] += (kpp_inverse ** (1 / num_words))
                                absent_bucket_samples[bucket_pos] += 1
                                if stem_text(kp_pred) in stem_absent_targets:
                                    absent_bucket_acc[bucket_pos] += 1

                        kpp_inverse = 1

                        collect_tokens = ''

                else:
                    #print(tokenizer.decode(idx_tokens[j]))
                    kpp_inverse *= t5_probs[i][j]
                    collect_tokens += token

            pbar.update(1)

    '''print(present_samples)
    print(absent_samples)
    print(present_bucket_samples)
    print(absent_bucket_samples)
    print(present_correct, present_bucket_acc, np.sum(present_bucket_acc))
    print(absent_correct, absent_bucket_acc, np.sum(absent_bucket_acc))
    print((present_bucket_confidence + absent_bucket_confidence) / (present_bucket_samples + absent_bucket_samples))
    '''

    print((present_bucket_acc + absent_bucket_acc) / (present_bucket_samples + absent_bucket_samples))

    ece, mce, tce = calculate_error(present_samples + absent_samples, present_bucket_samples + absent_bucket_samples,
                                    (present_bucket_confidence + absent_bucket_confidence) / (
                                                present_bucket_samples + absent_bucket_samples),
                                    (present_bucket_acc + absent_bucket_acc) / (
                                                present_bucket_samples + absent_bucket_samples))
    print('ECE:',ece)
    present_ece, present_mce, present_tce = calculate_error(present_samples, present_bucket_samples, present_bucket_confidence/present_bucket_samples, present_bucket_acc/present_bucket_samples)
    absent_ece, absent_mce, absent_tce = calculate_error(absent_samples, absent_bucket_samples, absent_bucket_confidence/absent_bucket_samples, absent_bucket_acc/absent_bucket_samples)
    print(f'present_ece: {present_ece}')
    print(f'absent_ece: {absent_ece}')

    print()
    return (present_bucket_acc + absent_bucket_acc) / (present_bucket_samples + absent_bucket_samples)




def calibrate_all_datasets(model = 't5'):
    datasets = [ 'semeval', 'krapivin','inspec', 'kp20k']#, 'kp20k']
    dict = {}
    for dataset in datasets:
        print(dataset, model)
        if model == 'exhird':
            dict[dataset] = calibrate_exhird(dataset, num_buckets=10).tolist()
        elif model == 't5':
            dict[dataset] = calibrate_t5(dataset, num_buckets=10).tolist()
        elif model == 'bart':
            dict[dataset] = calibrate_bart(dataset, num_buckets=10).tolist()
        elif model == 'one2seq':
            dict[dataset] = calibrate_one2set(dataset, num_buckets=10, model2=model+'_').tolist()
        else:
            dict[dataset] = calibrate_one2set(dataset, num_buckets=10, model2=model + '_').tolist()
    json_name = model+'_calibrate_kpp_values'
    #with open('data_dump/'+json_name+'.json', 'w') as f:
    #    json.dump(dict,f)



calibrate_all_datasets(model='exhird')
#plot_reliability('calibrate_kpp_values', plot_name='Calibration_new', num_buckets=10, model1='t5', model2='bart')

#calibrate_exhird('inspec', num_buckets=10)
#calibrate_t5('inspec', num_buckets=10)

