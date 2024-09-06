from generate_graphs import *
import tqdm
import pandas as pd
from generate_graphs_transformers import json_load_one2seq
from utilities.utils import *
def json_load_dump_t5(dataset, probab = False):
    with open('data_dump/t5_large/' + dataset + '_data.json', 'r') as f:
        dic = json.load(f)

    scores = dic['perplexities']

    kp_predictions = dic['predictions']
    #print(kp_predictions[0])
    probabilities = []#dic['probabilities']
    predicted_tokens = dic['token_predictions']
    print(predicted_tokens[0])
    context_lines = dic['src']


    for i, pred in enumerate(kp_predictions):

        j=0
        while j< len(pred):
            if pred[j] == '':
                pred.remove('')
            else:
                j+=1
        if len(pred) != len(scores[i]):
            scores[i] = scores[i][:len(pred)]
    if probab == True:
        return scores, kp_predictions, context_lines, probabilities, predicted_tokens
    else:
        return scores, kp_predictions, context_lines


def get_t5_targets(dataset):

    path = 'processed_data/KG_test_'+dataset+'.jsonl'
    targets = []
    with jsonlines.open(path, "r") as Reader:
        for id, obj in enumerate(Reader):
            targets.append(obj['trg'].split(';'))
    return targets




def get_transformers_ppl(predictions, scores, context_lines, do_search=False):

    present_ppl, absent_ppl = [], []

    for i, pred in enumerate(predictions):
        context = ' '.join([stemmer.stem(w) for w in context_lines[i].strip().split()])

        for j, kp in enumerate(pred):
            keyphrase  = ' '.join([stemmer.stem(w) for w in kp.split()])

            if keyphrase in context:
                present_ppl.append(scores[i][j])
            else:
                absent_ppl.append(scores[i][j])


    return present_ppl, absent_ppl

def remove_duplicates(scores, predictions):
    for i, pred in enumerate(predictions):
        score_pred = []
        set_pred = []
        for j, kp in enumerate(pred):
            if kp not in set_pred:
                score_pred.append(scores[i][j])
                set_pred.append(predictions[i][j])
        scores[i] = score_pred
        predictions[i] = set_pred
    return scores, predictions


def plot_histogram_transformers():

    datasets = ['kp20k','krapivin', 'inspec', 'semeval']
    bins_num=30
    min_lim = 1
    max_lim = 4
    linewidth = 1.5
    font_size_extra=14
    font_size = 12
    font_size_labels = 10
    color = sns.color_palette("bright")
    j=1
    fig = plt.figure(figsize=[14, 13])
    fig.text(0.001, 0.55, 'Count', va='center', rotation='vertical', fontsize=font_size_extra)

    fig.text(0.45, 0.015, 'Keyphrase perplexity', va='center', fontsize=font_size_extra)
    for i, dataset in enumerate(datasets):
        model1 = 'exhird_h_'
        scores, predictions, entropies = json_load(model1, dataset.lower())
        present_ppl, absent_ppl = get_ppl(predictions, scores, model1)

        axes1 = plt.subplot(4,2,j)

        if i==0:
            axes1.set_title('ExHiRD', fontsize = font_size)

        j+=1
        axes1 = sns.distplot(present_ppl, bins=bins_num,
                             hist_kws={'range': [min_lim, max_lim]},
                             hist=True,
                             kde=False,
                             color=color[1],
                             label="present"
                             )
        if dataset == 'kp20k':
            name = 'KP20k'
        else:
            name = dataset.capitalize()
        axes1.set_xlabel(name, fontsize= font_size)
        axes1.autoscale(enable=True, axis='x', tight=True)
        bottom, top = axes1.get_ylim()
        #axes1.set_xlabel('Perplexities')
        axes1 = sns.distplot(absent_ppl, bins=bins_num, hist_kws={'range': [min_lim, max_lim]},
                             hist=True,
                             kde=False,
                             color=color[2],
                             label="absent"
                             )
        if i == 0:
            axes1.legend(frameon=False, prop={'size': 10}, loc='upper right')

        axes1.tick_params(labelsize=font_size_labels)
        plt.axvline(statistics.median(present_ppl), color=color[1], linestyle='dashed', linewidth=linewidth)
        plt.axvline(statistics.median(absent_ppl), color=color[2], linestyle='dashed', linewidth=linewidth)


        print(f"{statistics.median(present_ppl), statistics.median(absent_ppl),}")

        plt.setp(axes1.get_xticklabels(), visible=True)

        model2 = 't5_'
        scores, predictions, context_lines = json_load_dump(dataset.lower())
        scores, predictions = remove_duplicates(scores, predictions)
        present_ppl, absent_ppl = get_transformers_ppl(predictions, scores, context_lines)

        #present_ppl, absent_ppl= normalize(present_ppl, absent_ppl)

        with sns.color_palette("Set2"):
            axes2 = plt.subplot(4,2,j, sharey=axes1)

            if i==0:
                axes2.set_title('T5', fontsize = font_size)
            j+=1


            axes2.set_xlabel(name, fontsize = font_size)
            axes2 = sns.distplot(present_ppl, bins=bins_num,
                                 hist_kws={'range': [min_lim, max_lim], }, hist=True, kde=False,
                                 label="present",
                                 color=color[1]
                                 )
            axes2 = sns.distplot(absent_ppl, bins=bins_num,
                                 hist_kws={'range': [min_lim, max_lim], }, hist=True, kde=False,
                                 label="absent",
                                 color=color[2]
                                 )

            #statistics.median(present_ppl)
            plt.axvline(statistics.median(present_ppl), color=color[1], linestyle='dashed', linewidth=linewidth)
            plt.axvline(statistics.median(absent_ppl), color=color[2], linestyle='dashed',
                        linewidth=linewidth)

            axes2.tick_params(labelsize=font_size_labels)
            axes2.autoscale(enable=True, axis='x', tight=True)
            axes2.set_ylim(bottom=bottom, top=top)

    fig.tight_layout(pad=1.5)


    plt.savefig('graphs/perplexities_thesis_new.png')
    plt.show()
    plt.close()


#plot_histogram_transformers()

def stem_text(text):
    return ' '.join([stemmer.stem(w) for w in text.strip().split()])



def search_para(pred, context):
    if pred in context:
        return False
    pred = pred.split()
    for i, word in enumerate(pred):
        if word not in context:
            return False
    return True


def get_relative_ppl_transformers(predictions, scores, context_lines):
    relative_ppl = [[] for i in range(5)]
    bins = [0.2, 0.4, 0.6, 0.8, 1.0]

    for i, pred in enumerate(predictions):

        context = ' '.join([stemmer.stem(w) for w in context_lines[i].strip().split()])

        for j, keyphrase in enumerate(pred):
            try:
                pos = context.index(' '.join([stemmer.stem(w) for w in keyphrase.split()]))
                relative_pos = pos / float(len(context))


                for k, bin in enumerate(bins):

                    if relative_pos < bin:
                        ind = k
                        break
                relative_ppl[ind].append(scores[i][j])
            except:
                continue
    return relative_ppl




def plot_sentence_pos_transformers(dataset):


    #preds_file = open(opt.output, encoding='utf-8')
    #preds_lines = preds_file.readlines()

    model1 = 't5_'
    scores, predictions, context_lines = json_load_dump(dataset)
    relative_ppl1 = get_relative_ppl_transformers(predictions, scores, context_lines)
    make_boxplot(relative_ppl1, model1 + dataset, 'Relative pos', 'Perplexity', model1 + dataset)


    model2 = 'exhird_h_'
    src = 'data/test_datasets/processed_' + dataset + '_testing_context.txt'
    context_file = open(src, encoding='utf-8')
    context_lines = context_file.readlines()

    scores, predictions, entropies = json_load(model2, dataset)
    relative_ppl2 = get_relative_ppl(predictions, scores, context_lines, model2)

    make_boxplot(relative_ppl2, model2 + dataset, 'Relative pos', 'Perplexity', model2 + dataset)



    keys = ['Relative Position', 'Perplexity']

#plot_sentence_pos_transformers('kp20k')



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
    target_file = open(path, encoding='utf-8')
    target_lines = target_file.readlines()
    targets = []
    for i, preds in enumerate(target_lines):
        preds = preds.split(';')
        for j, pred in enumerate(preds):
            preds[j] = pred.strip()
        targets.append(preds)
    return targets

def clean_one2set_preds_for_relative_pos(predictions):
    clean_preds = []
    for preds in predictions:
        new_preds = []
        tmp_kp = ''
        for i, token in enumerate(preds):
            if token != '<sep>':
                if len(tmp_kp)==0:
                    tmp_kp+=token
                else:
                    tmp_kp += ' '+token
            else:
                if len(tmp_kp)>0:
                    tmp_kp = tmp_kp.strip()
                    if tmp_kp not in new_preds:
                        new_preds.append(tmp_kp)
                    tmp_kp=''

        clean_preds.append(new_preds)
    return clean_preds

def get_relative_error_numbers(model, dataset):
    '''
    model1 = 'exhird_h_'
    scores1, predictions1, entropies = json_load(model1, dataset)

    model2 = 't5_'
    scores2, predictions2, context_lines = json_load_dump(dataset)
    scores2, predictions2 = remove_duplicates(scores2, predictions2)
    targets = get_target(dataset)
    '''

    if model=='t5':
        scores, predictions, context_lines = load_t5_preds(dataset)
        scores, predictions = remove_duplicates(scores, predictions)
        targets = get_t5_targets(dataset)
    elif model == 'exhird_h_':
        scores, predictions, entropies = json_load(model, dataset)
        context_lines = get_exhird_context(dataset)
        predictions, _ = format_exhird_predictions(predictions, scores)
        targets = get_exhird_targets(dataset)
    elif model == 'one2seq_':
        scores, raw_predictions, context_lines = json_load_one2seq(model, dataset.lower())
        targets = read_one2seq_targets(dataset)
        # present_ppl, absent_ppl = get_one2seq_ppl(raw_predictions, prob_scores, context_lines)
        predictions = preprocess_one2seq_predictions_to_kps(raw_predictions)
    elif model == 'bart':
        context_lines, predictions, probs, predicted_tokens, targets = load_bart_preds(dataset)

        probs, predictions = remove_duplicates(probs, predictions)

    else:
        scores, predictions, context_lines = json_load_one2set(model, dataset.lower())
        predictions = clean_one2set_preds_for_relative_pos(predictions)
        targets = read_one2seq_targets(dataset)




    relative_errors = [0 for i in range(5)]
    bins = [0.2, 0.4, 0.6, 0.8, 1.0]

    total_kp = [0 for i in range(5)]
    total_kp_num = 0

    for i, context in enumerate(context_lines):

        if model == 'exhird_h_':
            stem_context = stem_text(context[0])
        else:
            stem_context = stem_text(context)
        #print(context_lines[i])
        if type(context_lines[i]) == list:
            context_lines[i] = context_lines[i][0]
        targets[i], absent_targets = segregate_kps(targets[i], context_lines[i])
        stemmed_pred = [stem_text(kp) for kp in predictions[i]]
        print(stemmed_pred)


        for j, keyphrase in enumerate(targets[i]):

            #if dataset == 'semeval':
            #    stem_kp = keyphrase
            #else:
            #    stem_kp = stem_text(keyphrase)
            #print(stem_kp)
            stem_kp = keyphrase
            if stem_kp not in stem_context:
                continue
            total_kp_num += 1
            pos = stem_context.index(stem_kp)
            relative_pos = pos / float(len(stem_context))
            #print(relative_pos)

            for k, bin in enumerate(bins):

                if relative_pos < bin:
                    ind = k
                    break

            total_kp[ind] +=1

            if stem_kp not in stemmed_pred:
                relative_errors[ind] +=1

        #print()
    #print(total_kp_num)
    #print(total_kp)
    error_list = [ relative_errors[i]/float(total_kp[i]) for i in range(5)]

    return error_list


def plot_relative_pos_graph(model1 = 'exhird_h_', model2='t5'):
    datasets = ['kp20k', 'krapivin', 'inspec', 'semeval']
    percentages = ['0-20', '20-40', '40-60', '60-80', '80-100']


    font_size = 13
    font_size_labels = 11
    color = sns.color_palette("pastel")
    j = 1
    fig = plt.figure(figsize=[7, 5])
    fig.text(0.001, 0.5, 'Error percentage', va='center', rotation='vertical', fontsize=font_size)

    fig.text(0.4, 0.025, 'Positional range', va='center', fontsize=font_size)

    X = np.array([0.06*i for i in range(5)])
    for i, dataset in enumerate(datasets):

        #exhird_errors, t5_errors = get_relative_error_numbers(dataset)
        exhird_errors = get_relative_error_numbers(model1, dataset.lower())
        t5_errors = get_relative_error_numbers(model2, dataset.lower())
        axes1 = plt.subplot(2, 2, j)



        #ax = fig.add_axes([0, 0, 1, 1])
        width = 0.02
        j += 1
        axes1.bar(X , exhird_errors, color=color[1], width=width, label = 'ExHiRD', edgecolor='black')
        axes1.bar(X + width, t5_errors, color=color[2], width=width, label = 'One2Seq', edgecolor='black')
        if dataset[0].islower():
            name = dataset.capitalize()
        else:
            name = dataset
        axes1.set_xlabel(name, fontsize=font_size)
        axes1.autoscale(enable=True, axis='x', tight=True)
        plt.ylim(bottom=0, top=1.05)
        plt.xticks(X + width / 2, tuple(percentages))
        plt.grid(axis='y', color='gray', linestyle = 'dashed', alpha=0.4)
        plt.grid(axis='y', color='gray', linestyle = 'dashed', alpha=0.4)
        plt.grid(axis='y', color='gray', linestyle = 'dashed', alpha=0.4)
        if i == 0:
            axes1.legend(frameon=False, prop={'size': 8})


        axes1.tick_params(labelsize=font_size_labels)
        #print(i)
        #plt.setp(axes1.get_xticklabels(), visible=True)



    fig.tight_layout(pad=1.5)

    plt.savefig('graphs/relative_pos_comparison_all_datasets_'+model1+'_'+model2+'.png')
    plt.show()
    plt.close()

#plot_relative_pos_graph(model2='one2seq')

def correct_dig_text(text):
    text= text.split()
    for i, token in enumerate(text):
        if token.isalpha():
            continue
        else:
            text[i] = '<digit>'
    return ' '.join(text)


def doc_search(dataset, correct_dig=True):
    model1 = 'exhird_h_'
    scores, predictions, entropies = json_load(model1, dataset)
    exhird_context_lines = get_exhird_context(dataset)
    exhird_predictions, _ = format_exhird_predictions(predictions, scores)
    exhird_targets = get_exhird_targets(dataset)

    model2 = 't5_'
    scores, predictions, t5_context_lines = json_load_dump(dataset)
    scores, t5_predictions = remove_duplicates(scores, predictions)
    t5_targets = get_t5_targets(dataset)
    consensus =0
    total_kp = 0
    bins = 3
    total_kp_details = [0 for i in range(bins)]
    present_total_consensus_details = [0 for i in range(bins)]
    absent_total_consensus_details = [0 for i in range(bins)]
    present_consensus_found = [0 for i in range(bins)]
    absent_consensus_found = [0 for i in range(bins)]
    consensus_collect = []
    exhird_details = [0 for i in range(bins)]
    t5_details = [0 for i in range(bins)]
    present_union = [0 for i in range(bins)]
    absent_union=[0 for i in range(bins)]

    look_up_vector = [0 for i in range(len(t5_context_lines))]


    for i in tqdm.tqdm(range(len(exhird_context_lines))):

        context1 = exhird_context_lines[i][0]
        flag = 0
        #print(i, consensus, total_kp)
        high_percent = -1
        total_kp += len(exhird_targets[i])

        for pos, kp in enumerate(exhird_targets[i]):
            if correct_dig:
                exhird_targets[i][pos] = correct_dig_text(kp)
            kp = kp.split()
            if len(kp)>=bins:
                total_kp_details[-1] +=1
            else:
                total_kp_details[len(kp)-1] +=1
        #print(context1)
        if correct_dig:
            context1 = correct_dig_text(context1)


        for j, context2 in enumerate(t5_context_lines):
            #set1=set(stem_text(context1).split())
            #set2=set(stem_text(context2).split())

            #common_words = list(set1&set2)
            #percent = len(common_words)/len(list(set(stem_text(context1).split(' '))))

            #if look_up_vector[j] ==1:
            #    continue

            #percent = len(set(t5_targets[j])&set(exhird_targets[i]))/len(exhird_targets[i])
            #if percent > high_percent:
            #    high_percent=percent
            cnt =0
            for target in exhird_targets[i]:
                if target in t5_targets[j]:
                    cnt+=1
            without_digit_t5_targets = [kp for kp in t5_targets[j] if '<digit>' not in kp]

            if cnt>len(t5_targets[j])//2:#percent>0.5:
                #print(exhird_targets[i])
                #print(t5_targets[j])
                #print()
                #print(i, j)
                #look_up_vector[j] =1
                stem_context1 = stem_text(context1)
                stem_context2 = stem_text(context2)
                present_exhird_predictions = []
                absent_exhird_predictions = []
                present_t5_predictions = []
                absent_t5_predictions = []
                for kp in exhird_predictions[i]:
                    if stem_text(kp) in stem_context1:
                        present_exhird_predictions.append(kp)
                    else:
                        absent_exhird_predictions.append(kp)

                    kp = kp.split()
                    if len(kp)>=bins:
                        exhird_details[-1]+=1
                    else:
                        exhird_details[len(kp)-1] +=1
                for kp in t5_predictions[j]:
                    if stem_text(kp) in stem_context2:
                        present_t5_predictions.append(kp)
                    else:
                        absent_t5_predictions.append(kp)

                    kp = kp.split()
                    if len(kp)>=bins:
                        t5_details[-1]+=1
                    else:
                        t5_details[len(kp)-1] +=1
                for kp in list(set(present_exhird_predictions+present_t5_predictions)):
                    kp = kp.split()
                    if len(kp)>=bins:
                        present_union[-1] +=1
                    else:
                        present_union[len(kp)-1] +=1
                for kp in list(set(absent_t5_predictions+absent_exhird_predictions)):
                    kp=kp.split()
                    if len(kp)>=bins:
                        absent_union[-1] +=1
                    else:
                        absent_union[len(kp)-1]+=1


                common_pred = list(set(exhird_predictions[i])&set(t5_predictions[j]))
                consensus_collect.append([common_pred, exhird_targets[i], context1])
                for k, pred in enumerate(common_pred):

                    tmp_pred = pred.split()

                    if len(tmp_pred)>=bins:
                        if pred in context1:
                            present_total_consensus_details[-1] += 1
                        else:
                            absent_total_consensus_details[-1] +=1
                    else:
                        if pred in context1:
                            present_total_consensus_details[len(tmp_pred)-1] += 1
                        else:
                            absent_total_consensus_details[len(tmp_pred)-1] +=1


                    stem_pred = stem_text(pred)
                    if stem_pred in [stem_text(kp) for kp in exhird_targets[i]]:
                        consensus+=1
                        if len(tmp_pred) >=bins:
                            if pred in context1:
                                present_consensus_found[-1] +=1
                            else:
                                absent_consensus_found[-1] +=1
                        else:
                            if pred in context1:
                                present_consensus_found[len(tmp_pred)-1] +=1
                            else:
                                absent_consensus_found[len(tmp_pred)-1] +=1

                flag=1
                break

        if flag==0:
            print('Not found! '+str(high_percent)+' '+context1 )
            print(exhird_targets[i])
            consensus_collect.append([])



    dic = {
        'consensus_not_found': consensus,
        'total_kp': total_kp,
        'total_kp_details' : total_kp_details,
        'present_total_consensus_details': present_total_consensus_details,
        'absent_total_consensus_details': absent_total_consensus_details,
        'present_consensus_found_in_gold': present_consensus_found,
        'absent_consensus_found_in_gold': absent_consensus_found,
        'exhird_pred_details':exhird_details,

        't5_pred_details': t5_details,
        'present_union': present_union,
        'absent_union': absent_union,
        'consensus_collect': consensus_collect

    }
    for key in list(dic.keys())[:-1]:
        print(key, dic[key])
    with open('stats/'+dataset+'_consensus_stats.json', 'w') as f:
        json.dump(dic,f)


def get_consensus_stats(dataset):
    doc_search(dataset, False)
    with open('stats/'+dataset+'_consensus_stats.json', 'r') as f:
        data = json.load(f)
    mc_present = []
    mc_absent = []
    mcg_present = []
    mcg_absent = []
    for i, stat in enumerate(data['present_total_consensus_details']):

        mc_present.append(data['present_total_consensus_details'][i] *100 / data['present_union'][i])

        mc_absent.append(data['absent_total_consensus_details'][i] *100 / data['absent_union'][i])
        try:
            mcg_present.append(data['present_consensus_found_in_gold'][i]*100 / data['present_total_consensus_details'][i])
        except:
            mcg_present.append(0)
        try:
            mcg_absent.append(data['absent_consensus_found_in_gold'][i] * 100 / data['absent_total_consensus_details'][i])
        except:
            mcg_absent.append(0)
    print(mc_present)
    print(mc_absent)
    print(mcg_present)
    print(mcg_absent)

#get_consensus_stats('kp20k')


def get_flattened_ppl_with_truth(dataset_predictions, ppl, targets):
    truth_vector, ppl_flat = [], []
    for j, predictions in enumerate(dataset_predictions):
        for k, kp in enumerate(predictions):
            ppl_flat.append(ppl[j][k])
            if stem_text(kp) in targets[j]:
                truth_vector.append(1)
            else:
                truth_vector.append(0)
    return truth_vector, ppl_flat

def sort_ppl_with_truth(ppl, truth):
    ppl_flat, truth_vector= [],[]
    for x,y in sorted(zip(ppl, truth)):
        ppl_flat.append(x)
        truth_vector.append(y)
    return ppl_flat, truth_vector

def error_values(ppl_vector, truth_vector, min_lim, max_lim, interval):

    error_counts = []

    while min_lim<max_lim:
        total = 0
        errors = 0
        for i, val in enumerate(ppl_vector):
            if val >= min_lim and val < max_lim:
                total +=1
                if truth_vector[i] == 1:
                    errors+=1
            if val > max_lim:
                break
        error_counts.append(errors)
        min_lim+=interval
    return error_counts


def ppl_vs_error():

    datasets = ['kp20k', 'krapivin', 'inspec', 'semeval']



    font_size = 13
    font_size_labels = 11
    color = sns.color_palette("bright")
    j = 1
    fig = plt.figure(figsize=[7, 6])
    fig.text(0.001, 0.5, 'Correct predictions count', va='center', rotation='vertical', fontsize=font_size)
    min_lim = 1
    max_lim = 5
    interval= 0.2

    fig.text(0.5, 0.025, 'Perplexity', va='center', fontsize=font_size)

    #exhird_ppl_all, exhird_truth_all = [], []
    #t5_ppl_all, t5_truth_all = [], []
    X = np.array([min_lim + i*interval - (interval/2) for i in range(1, int((max_lim-min_lim)/interval)+1)])
    X_label_intervals = np.array([min_lim + i * interval  for i in range(0, int((max_lim - min_lim) / interval),2)])
    for i, dataset in enumerate(datasets):
        scores, predictions, t5_context_lines = json_load_dump(dataset.lower())
        t5_ppl, t5_predictions = remove_duplicates(scores, predictions)
        t5_targets = get_t5_targets(dataset)

        scores, predictions, entropies = json_load('exhird_h_', dataset)
        exhird_context_lines = get_exhird_context(dataset)
        exhird_predictions, exhird_ppl = format_exhird_predictions(predictions, scores)
        exhird_targets = get_exhird_targets(dataset)

        t5_truth, t5_ppl_flat = get_flattened_ppl_with_truth(t5_predictions, t5_ppl, t5_targets)
        t5_ppl_flat, t5_truth = sort_ppl_with_truth(t5_ppl_flat, t5_truth)
        exhird_truth, exhird_ppl_flat = get_flattened_ppl_with_truth(exhird_predictions, exhird_ppl, exhird_targets)
        exhird_ppl_flat, exhird_truth = sort_ppl_with_truth(exhird_ppl_flat, exhird_truth)

        '''
        
        exhird_ppl_all.append(exhird_ppl_flat)
        exhird_truth_all.append(exhird_truth)
        t5_ppl_all.append(t5_ppl_flat)
        t5_truth_all.append(t5_truth)
        

        exhird_ppl_all += exhird_ppl_flat
        exhird_truth_all += exhird_truth
        t5_ppl_all += t5_ppl_flat
        t5_truth_all += t5_truth
        '''

        exhird_counts = error_values(exhird_ppl_flat, exhird_truth, min_lim, max_lim, interval)
        t5_counts = error_values(t5_ppl_flat, t5_truth, min_lim, max_lim, interval)


        axes1 = plt.subplot(2,2,j)
        j+=1

        axes1= sns.lineplot(X, exhird_counts, color=color[2], label='ExHiRD', legend=False, marker='s')
        axes1 = sns.lineplot(X, t5_counts, color=color[1],  label='T5', legend=False, dashes=True, marker='X')

        if dataset == 'kp20k':
            name = 'KP20k'
        else:
            name = dataset.capitalize()

        axes1.set_xlabel(name, fontsize=font_size)
        axes1.autoscale(enable=True, axis='x', tight=True)

        plt.xticks(X_label_intervals)
        plt.grid(axis='y', color='gray', linestyle='dashed', alpha=0.4)

        if i == 0:
            axes1.legend(frameon=False, prop={'size': font_size_labels})

        axes1.tick_params(labelsize=font_size_labels-1)

        # plt.setp(axes1.get_xticklabels(), visible=True)

    fig.tight_layout(pad=1.5)

    plt.savefig('graphs/correct_predictions_count.png')
    plt.show()
    plt.close()


#ppl_vs_error()

def make_sns_boxplot(exhird_bins, t5_bins, filename, xlabel, ylabel, title):
    sns.set_theme(style="whitegrid", font_scale=1.3)

    color = sns.color_palette("pastel")
    j = 1
    fig = plt.figure(figsize=[9, 5])

    fig.text(0.001, 0.5, 'Probability', va='center', rotation='vertical', fontsize=16)
    fig.text(0.45, 0.01, 'Token position', va='center', fontsize=16)
    axes1 = plt.subplot(1, 2, 1)
    dic = {
        'probability' : [],
        'legend' : [],
        'token_pos' : []
    }
    p_bins, a_bins = exhird_bins
    for i, bin in enumerate(p_bins):
        for j, probability in enumerate(bin):
            dic['probability'].append(probability)
            dic['legend'].append('present')
            dic['token_pos'].append(i+1)

    for i, bin in enumerate(a_bins):
        for j, probability in enumerate(bin):
            dic['probability'].append(probability)
            dic['legend'].append('absent')
            dic['token_pos'].append(i+1)
    df= pd.DataFrame(data=dic)
    axes1 = sns.boxplot(x="token_pos", y="probability", hue="legend", data=df, showfliers= False,
                        palette=[color[1], color[2]]).set(xlabel=None, ylabel=None, title='ExHiRD')
    plt.legend(frameon=False, prop={'size': 9}, loc=(0.003,0.91), ncol=2)
    plt.ylim(bottom=0, top=1.1)
    dic = {
        'probability': [],
        'legend': [],
        'token_pos': []
    }
    p_bins, a_bins = t5_bins
    axes2 = plt.subplot(1, 2, 2)
    for i, bin in enumerate(p_bins):
        for j, probability in enumerate(bin):
            dic['probability'].append(probability)
            dic['legend'].append('present')
            dic['token_pos'].append(i + 1)

    for i, bin in enumerate(a_bins):
        for j, probability in enumerate(bin):
            dic['probability'].append(probability)
            dic['legend'].append('absent')
            dic['token_pos'].append(i + 1)
    df = pd.DataFrame(data=dic)
    axes2 = sns.boxplot(x="token_pos", y="probability", hue="legend", data=df, showfliers=False,
                        palette=[color[1], color[2]]).set(xlabel=None, ylabel=None, title='T5')



    #axes1.legend(frameon=False, prop={'size': 9}, loc='upper left')
    #plt.xticks([i + 1 for i in np.arange(len(data))])
    plt.legend([], [], frameon=False)




    plt.ylim(bottom=0, top = 1.1)
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    #plt.title(title)
    #ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('graphs/'+filename + '.png', bbox_inches = 'tight', pad_inches = 0.02)
    plt.clf()

def probab_exhird_boxplots(dataset):
    model1 = 'exhird_h_'
    scores, predictions, entropies = json_load(model1, dataset)

    #make_boxplot(relative_ppl1, model1 + dataset, 'Relative pos', 'Perplexity', model1 + dataset)
    p_bins = [[] for i in range(5)]
    a_bins = [[] for i in range(5)]
    kp_len = 0
    kp_type = ''

    for i, pred in enumerate(predictions):
        for j, token in enumerate(pred):

            if token == '<p_start>':
                kp_type = 'present'

            elif token == '<a_start>':
                kp_type = 'absent'

            elif token == ';':
                kp_len=0
            else:
                if kp_len==4:
                    print(4)
                if kp_type == 'present':

                    if kp_len<5:
                        p_bins[kp_len].append(scores[i][j])
                    #else:
                    #    p_bins[-1].append(scores[i][j])
                else:
                    if kp_len<5:
                        a_bins[kp_len].append(scores[i][j])
                    #else:
                    #    a_bins[-1].append(scores[i][j])
                kp_len+=1

    #make_sns_boxplot(p_bins, a_bins, model1 + dataset +'_present_absent_', 'Token position', 'Probability', '')
    #print('Generated plots')
    return p_bins, a_bins
    #make_boxplot(a_bins, model1 + dataset+'_absent', 'Token position', 'Absent Probability', model1 + dataset)
#probab_exhird_boxplots('kp20k')
