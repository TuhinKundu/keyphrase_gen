from generate_graphs import *
import tqdm
import pandas as pd
from utilities.post_process_utils import *

def json_load_dump(dataset, probab = False, seed=1):
    with open('data_dump/seed'+str(seed)+'/' + dataset + '_data.json', 'r') as f:
        dic = json.load(f)

    ppl = dic['perplexities']

    kp_predictions = dic['predictions']
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


def get_t5_targets(dataset):

    path = 'processed_data/KG_test_'+dataset+'.jsonl'
    targets = []
    with jsonlines.open(path, "r") as Reader:
        for id, obj in enumerate(Reader):
            targets.append(obj['trg'].split(';'))
    return targets




def get_transformers_ppl(predictions, scores, context_lines):

    present_ppl, absent_ppl = [], []
    #print(predictions)
    for i, pred in enumerate(predictions):
        context = ' '.join([stemmer.stem(w) for w in context_lines[i].strip().split()])

        for j, kp in enumerate(pred):
            #print(kp)
            keyphrase  = ' '.join([stemmer.stem(w) for w in kp.split()])


            if keyphrase in context:
                present_ppl.append(scores[i][j])
            else:
                absent_ppl.append(scores[i][j])

    exit()
    return present_ppl, absent_ppl

def remove_duplicates_(scores, predictions):
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

def get_one2seq_ppl(predictions, scores, context_lines):
    present_ppl, absent_ppl = [], []

    for i, pred in enumerate(predictions):
        stemmed_context = stem_text(context_lines[i])
        kp_collect = []
        prob_collect = []
        for j, token in enumerate(pred):

            if token == "<sep>":

                if len(kp_collect) > 0:
                    ppl = np.prod(prob_collect) ** (-1 / float(len(prob_collect)))
                    stemmed_kp = stem_text(' '.join(kp_collect))
                    #print(prob_collect)
                    #print(kp_collect)
                    #print(ppl)
                    if stemmed_kp in stemmed_context:
                        present_ppl.append(ppl)
                    else:
                        #print(stemmed_kp)
                        #print(stemmed_context)
                        #print(pred)
                        #print()
                        absent_ppl.append(ppl)
                    kp_collect, prob_collect = [], []
            else:
                if len(pred)!=len(scores[i]):
                    print(kp_collect, len(pred))
                    break
                kp_collect.append(token)
                prob_collect.append(scores[i][j])
        if len(kp_collect) > 0:
            ppl = np.prod(prob_collect) ** (-1 / float(len(prob_collect)))
            stemmed_kp = stem_text(' '.join(kp_collect))
            if stemmed_kp in stemmed_context:
                present_ppl.append(ppl)
            else:
                absent_ppl.append(ppl)
    return present_ppl, absent_ppl

def json_load_one2seq(model, dataset):
    with open('graph_outputs/'+model+dataset+'_all_output.json', 'r') as f:
        dic= json.load(f)

    scores = dic['scores']
    predictions = dic['predictions']
    #entropies = dic['entropies']
    context_lines = dic['context_lines']

    return scores, predictions,  context_lines


def plot_histogram_transformers():

    datasets = ['semeval']#'kp20k 'kp20k','krapivin', 'inspec',
    bins_num=30
    min_lim = 1
    max_lim = 6
    linewidth = 1.5
    font_size_extra=14
    font_size = 12
    font_size_labels = 10
    color = sns.color_palette("bright")
    j=1
    fig = plt.figure(figsize=[7, 6.5])
    fig.text(0.001, 0.55, 'Count', va='center', rotation='vertical', fontsize=font_size_extra)

    fig.text(0.35, 0.015, 'Keyphrase perplexity', va='center', fontsize=font_size_extra)
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


        print(f"{model1, statistics.median(present_ppl), statistics.median(absent_ppl)}")

        plt.setp(axes1.get_xticklabels(), visible=True)

        #model2 = 't5_'
        #scores, predictions, context_lines = json_load_dump(dataset.lower())
        #scores, predictions = remove_duplicates(scores, predictions)
        #present_ppl, absent_ppl = get_transformers_ppl(predictions, scores, context_lines)

        model2 = 'one2seq_'
        scores, predictions, entropies, context_lines = json_load_one2seq(model2, dataset.lower())
        present_ppl, absent_ppl = get_one2seq_ppl(predictions, scores, context_lines)

        #was orig commented
        #present_ppl, absent_ppl= normalize(present_ppl, absent_ppl)

        with sns.color_palette("Set2"):
            axes2 = plt.subplot(4,2,j, sharey=axes1)

            if i==0:
                axes2.set_title('One2Seq', fontsize = font_size)
            j+=1


            axes2.set_xlabel(name, fontsize = font_size)
            axes2.autoscale(enable=True, axis='x', tight=True)
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
            
            axes2.set_ylim(bottom=bottom, top=top)
            print(f"{model2, statistics.median(present_ppl), statistics.median(absent_ppl)}")

    fig.tight_layout(pad=1.5)


    plt.savefig('graphs/perplexities_'+model1+'_'+model2+'new.png')
    plt.show()
    plt.close()


#plot_histogram_transformers()





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
        scores, predictions, context_lines = json_load_dump(dataset)
        scores, predictions = remove_duplicates(scores, predictions)
        targets = get_t5_targets(dataset)
    else:
        scores, predictions, entropies = json_load(model, dataset)
        context_lines = get_exhird_context(dataset)
        predictions, _ = format_exhird_predictions(predictions, scores)
        targets = get_exhird_targets(dataset)


    relative_errors = [0 for i in range(5)]
    bins = [0.2, 0.4, 0.6, 0.8, 1.0]

    total_kp = [0 for i in range(5)]
    total_kp_num = 0

    for i, context in enumerate(context_lines):

        if model == 't5':
            stem_context = stem_text(context)
        else:
            stem_context = stem_text(context[0])
        stemmed_pred = [stem_text(kp) for kp in predictions[i]]

        for j, keyphrase in enumerate(targets[i]):

            stem_kp = stem_text(keyphrase)
            #print(stem_kp)
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

    return [ relative_errors[i]/float(total_kp[i]) for i in range(5)]


def get_partial_matches(model, dataset):

    if model=='t5':
        scores, predictions, context_lines = json_load_dump(dataset)
        scores, predictions = remove_duplicates(scores, predictions)
        targets = get_t5_targets(dataset)
    else:
        scores, predictions, entropies = json_load(model, dataset)
        context_lines = get_exhird_context(dataset)
        predictions, _ = format_exhird_predictions(predictions, scores)
        targets = get_exhird_targets(dataset)

    present_pred, absent_pred = 0,0
    partial_cnt = 0

    for i, context in enumerate(context_lines):

        if model == 't5':
            stem_context = stem_text(context)
        else:
            stem_context = stem_text(context[0])
        stemmed_pred = [stem_text(kp) for kp in predictions[i]]

        if dataset != 'semeval':
            stemmed_targets = [stem_text(target) for target in targets[i]]
        else:
            stemmed_targets = targets[i]

        #for j, stem_gold_kp in enumerate(stemmed_targets):
        j=0
        while j<len(stemmed_targets):
            stem_gold_kp = stemmed_targets[j]

            if stem_gold_kp in stemmed_pred:
                if stem_gold_kp in stem_context:
                    present_pred += 1

                else:
                    absent_pred += 1
                stemmed_targets.remove(stem_gold_kp)
                stemmed_pred.remove(stem_gold_kp)
            else:
                j+=1

        j=0
        check_partials = []
        while j<len(stemmed_targets):
            stem_gold_kp_split = stemmed_targets[j].split()

            if len(stem_gold_kp_split)>2:
                check_partials.append(' '.join(stem_gold_kp_split[1:-1]))

            if len(stem_gold_kp_split)>1:
                check_partials.append(' '.join(stem_gold_kp_split[1:]))
                check_partials.append(' '.join(stem_gold_kp_split[:-1]))
            j+=1
        print(targets[i])
        print(predictions[i])
        for j, partial in enumerate(check_partials):

            if partial in stemmed_pred:
                partial_cnt+=1
                print(partial)
        print()


    print(present_pred, absent_pred, partial_cnt)


#get_partial_matches('t5', 'inspec')


def plot_relative_pos_graph():
    datasets = [ 'kp20k','krapivin', 'inspec', 'semeval']
    percentages = ['0-20', '20-40', '40-60', '60-80', '80-100']


    font_size = 13
    font_size_labels = 11
    color = sns.color_palette("pastel")
    j = 1
    fig = plt.figure(figsize=[7, 5])
    fig.text(0.001, 0.5, 'Error percentage', va='center', rotation='vertical', fontsize=font_size)

    fig.text(0.4, 0.025, 'Positional range', va='center', fontsize=font_size)

    X = np.array([0.06*i for i in range(5)])
    for i, dataset in enumerate(datasets[1:]):

        #exhird_errors, t5_errors = get_relative_error_numbers(dataset)
        exhird_errors = get_relative_error_numbers('exhird_h_', dataset.lower())
        t5_errors = get_relative_error_numbers('t5', dataset.lower())
        axes1 = plt.subplot(2, 2, j)



        #ax = fig.add_axes([0, 0, 1, 1])
        width = 0.02
        j += 1
        axes1.bar(X , exhird_errors, color=color[1], width=width, label = 'ExHiRD', edgecolor='black')
        axes1.bar(X + width, t5_errors, color=color[2], width=width, label = 'T5', edgecolor='black')
        if dataset[0].islower():
            name = dataset.capitalize()
        else:
            name = dataset
        axes1.set_xlabel(name, fontsize=font_size)
        axes1.autoscale(enable=True, axis='x', tight=True)

        plt.xticks(X + width / 2, tuple(percentages))
        plt.grid(axis='y', color='gray', linestyle = 'dashed', alpha=0.4)
        plt.grid(axis='y', color='gray', linestyle = 'dashed', alpha=0.4)
        plt.grid(axis='y', color='gray', linestyle = 'dashed', alpha=0.4)
        if i == 0:
            axes1.legend(frameon=False, prop={'size': 8})


        axes1.tick_params(labelsize=font_size_labels)
        print(i)
        #plt.setp(axes1.get_xticklabels(), visible=True)



    fig.tight_layout(pad=1.5)

    plt.savefig('graphs/relative_pos_comparison_all_datasets_.png')
    plt.show()
    plt.close()



def correct_dig_text(text):
    text= text.split()
    for i, token in enumerate(text):
        if token.isalpha():
            continue
        else:
            text[i] = '<digit>'
    return ' '.join(text)





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

def make_sns_boxplot(exhird_bins, t5_bins, filename, xlabel, ylabel, title, model1='ExHiRD', model2='T5'):
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
                        palette=[color[1], color[2]]).set(xlabel=None, ylabel=None, title=model1)
    plt.legend(frameon=False, prop={'size': 12}, loc=(0.003,0.91), ncol=2)
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
                        palette=[color[1], color[2]]).set(xlabel=None, ylabel=None, title=model2)

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
                #if kp_len==4:
                #    print(4)
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



def probab_one2seq_boxplots(dataset):
    model2 = 'one2seq_'
    scores, predictions, entropies, context_lines = json_load_one2seq(model2, dataset.lower())
    #present_ppl, absent_ppl = get_one2seq_ppl(predictions, scores, context_lines)

    p_bins = [[] for i in range(5)]
    a_bins = [[] for i in range(5)]


    for i, pred in enumerate(predictions):
        stemmed_context = stem_text(context_lines[i])
        kp_collect = []
        prob_collect = []
        for j, token in enumerate(pred):
            if token == "<sep>":
                if len(kp_collect) > 0:
                    stemmed_kp = stem_text(' '.join(kp_collect))
                    if stemmed_kp in stemmed_context:
                        for num, prob in enumerate(prob_collect[:5]):
                            p_bins[num].append(prob)
                    else:
                        for num, prob in enumerate(prob_collect[:5]):
                            a_bins[num].append(prob)

                    kp_collect, prob_collect = [], []
            else:
                if len(pred) != len(scores[i]):
                    print(kp_collect, len(pred))
                    break
                kp_collect.append(token)
                prob_collect.append(scores[i][j])
        if len(kp_collect) > 0:

            stemmed_kp = stem_text(' '.join(kp_collect))
            if stemmed_kp in stemmed_context:
                for num, prob in enumerate(prob_collect[:5]):
                    p_bins[num].append(prob)
            else:
                for num, prob in enumerate(prob_collect[:5]):
                    a_bins[num].append(prob)

    exhird_bins= probab_exhird_boxplots(dataset)
    make_sns_boxplot(exhird_bins, [p_bins, a_bins], 'exhird_one2seq_boxplot_' + dataset + '_present_absent_', 'Token position',
                     'Probability', '', model2='One2Seq')


#probab_one2seq_boxplots('semeval')


from transformers import T5Tokenizer, BartTokenizer
from utilities.utils import load_t5_preds, load_bart_preds
def probab_t5_boxplots(dataset):

    model2 = 't5_'
    #ppl, t5_predictions, t5_context_lines, probabilites, token_predictions = json_load_dump(dataset, probab=True)

    ppl, t5_predictions, t5_context_lines, t5_probs, t5_tokens = load_t5_preds(dataset, probab=True)
    tokenizer = T5Tokenizer.from_pretrained('t5-tokenizer/')
    sep_token_id = tokenizer.encode("<sep>", add_special_tokens=False)[0]
    eos_token_id = tokenizer.eos_token_id

    p_bins = [[] for i in range(5)]
    a_bins = [[] for i in range(5)]



    for i, pred in enumerate(t5_tokens):

        kp_token_collect = []
        kp_probab_collect = []

        num_kp = 0

        kp_preds = set()

        for j, token_id in enumerate(pred[0]):

            if j==0:
                continue
            if token_id == sep_token_id or token_id == eos_token_id:

                if len(kp_probab_collect)>0:


                    kp_pred = tokenizer.decode(kp_token_collect)

                    stemmed_kp = stem_text(kp_pred)

                    if stemmed_kp in kp_preds:
                        kp_token_collect, kp_probab_collect = [], []
                        continue
                    else:
                        kp_preds.add(stemmed_kp)


                    if stemmed_kp in stem_text(t5_context_lines[i]):
                        #new bin calculation at keyphrase level
                        p_bins[num_kp].append(sum(kp_probab_collect)/float(len(kp_probab_collect)))

                        #old bin calculation token wise
                        #for k, bin in enumerate(kp_probab_collect[:5]):
                        #    p_bins[k].append(kp_probab_collect[k])
                    else:
                        #for k, bin in enumerate(kp_probab_collect[:5]):
                        #    a_bins[k].append(kp_probab_collect[k])
                        a_bins[num_kp].append((sum(kp_probab_collect))/float(len(kp_probab_collect)))


                    num_kp+=1
                    kp_token_collect = []
                    kp_probab_collect = []
                if token_id == eos_token_id or num_kp>4: #(0,1,2,3,4)
                    break
            else:
                kp_token_collect.append(token_id)
                kp_probab_collect.append(t5_probs[i][j])


        if len(kp_probab_collect) > 0:

            kp_pred = tokenizer.decode(kp_token_collect)

            stemmed_kp = stem_text(kp_pred)

            if stemmed_kp in kp_preds:
                continue

            if stemmed_kp in stem_text(t5_context_lines[i]):
                p_bins[num_kp].append(sum(kp_probab_collect)/float(len(kp_probab_collect)))
            else:
                a_bins[num_kp].append((sum(kp_probab_collect))/float(len(kp_probab_collect)))
    #exhird_bins = probab_exhird_boxplots(dataset)
    #print(p_bins)
    #print(a_bins)
    #make_sns_boxplot(exhird_bins, [p_bins, a_bins], 'final_boxplot_' + dataset + '_present_absent_', 'Token position', 'Probability', '')
    print(p_bins)
    return [p_bins, a_bins]

#probab_t5_boxplots('semeval')


def probab_bart_boxplots(dataset):
    t5_context_lines, t5_predictions, t5_probs, t5_tokens, t5_targets = load_bart_preds(dataset)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    sep_token = ' ;'
    eos_token = '</s>'

    p_bins = [[] for i in range(5)]
    a_bins = [[] for i in range(5)]

    for i, pred in enumerate(t5_tokens):

        kp_token_collect = []
        kp_probab_collect = []

        num_kp = 0
        #print(pred)

        kp_preds = set()
        for j, token_id in enumerate(pred):

            if token_id == sep_token or token_id == eos_token:

                if len(kp_probab_collect) > 0:

                    kp_pred = "".join(kp_token_collect).strip()
                    stemmed_kp = stem_text(kp_pred)

                    if stemmed_kp in kp_preds:
                        kp_token_collect, kp_probab_collect = [], []
                        continue
                    else:
                        kp_preds.add(stemmed_kp)

                    if stemmed_kp in stem_text(t5_context_lines[i]):
                        # new bin calculation at keyphrase level
                        p_bins[num_kp].append(sum(kp_probab_collect) / float(len(kp_probab_collect)))

                    else:
                        a_bins[num_kp].append((sum(kp_probab_collect)) / float(len(kp_probab_collect)))

                    num_kp += 1

                    kp_token_collect = []
                    kp_probab_collect = []
                if token_id == eos_token or num_kp > 4:  # (0,1,2,3,4)
                    break
            else:
                kp_token_collect.append(token_id)
                kp_probab_collect.append(t5_probs[i][j])

        if len(kp_probab_collect) > 0:

            kp_pred = "".join(kp_token_collect).strip()
            stemmed_kp = stem_text(kp_pred)

            if stemmed_kp in kp_preds:
                continue

            if stemmed_kp in stem_text(t5_context_lines[i]):
                p_bins[num_kp].append(sum(kp_probab_collect) / float(len(kp_probab_collect)))

            else:
                a_bins[num_kp].append((sum(kp_probab_collect)) / float(len(kp_probab_collect)))
    # exhird_bins = probab_exhird_boxplots(dataset)
    # print(p_bins)
    # print(a_bins)
    # make_sns_boxplot(exhird_bins, [p_bins, a_bins], 'final_boxplot_' + dataset + '_present_absent_', 'Token position', 'Probability', '')
    print(p_bins)
    return [p_bins, a_bins]
