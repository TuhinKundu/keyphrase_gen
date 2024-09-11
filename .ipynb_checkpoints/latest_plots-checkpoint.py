import matplotlib.pyplot as plt

from generate_graphs import *
import tqdm
import pandas as pd
from generate_graphs_transformers import json_load_one2seq
from utilities.utils import *
from generate_graphs_transformers_new import *
from generate_graphs_transformers import get_one2seq_ppl, probab_t5_boxplots, probab_bart_boxplots, json_load_dump
from transformers import T5Tokenizer
'''
def plot_histogram_transformers():
    datasets = ['kp20k', 'krapivin', 'inspec','semeval']  # 'kp20k 'kp20k',,
    bins_num = 30
    min_lim = 0
    max_lim = 5
    linewidth = 1.5
    font_size_extra = 14
    font_size = 10
    font_size_labels = 11
    color = sns.color_palette("bright")
    j = 1
    fig = plt.figure(figsize=[7, 6.5])
    fig.text(0.001, 0.55, 'Count', va='center', rotation='vertical', fontsize=font_size_extra)

    fig.text(0.35, 0.015, 'Keyphrase perplexity', va='center', fontsize=font_size_extra)
    for i, dataset in enumerate(datasets):
        model1 = 'exhird_h_'
        scores, predictions, entropies = json_load(model1, dataset.lower())
        present_ppl, absent_ppl = get_ppl(predictions, scores, model1)

        axes1 = plt.subplot(4, 3, j)

        if i == 0:
            axes1.set_title('ExHiRD', fontsize=font_size)

        j += 1
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
        axes1.set_xlabel(name, fontsize=font_size)
        axes1.autoscale(enable=True, axis='x', tight=True)
        bottom, top = axes1.get_ylim()
        # axes1.set_xlabel('Perplexities')
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

        model2 = 't5_'
        scores, predictions, context_lines = json_load_dump(dataset.lower())
        present_ppl, absent_ppl = get_transformers_ppl(predictions, scores, context_lines)
        print(present_ppl)
        with sns.color_palette("Set2"):
            axes2 = plt.subplot(4, 3, j, sharey=axes1)

            if i == 0:
                axes2.set_title('T5', fontsize=font_size)
            j += 1

            axes2.set_xlabel(name, fontsize=font_size)
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

            # statistics.median(present_ppl)
            plt.axvline(statistics.median(present_ppl), color=color[1], linestyle='dashed', linewidth=linewidth)
            plt.axvline(statistics.median(absent_ppl), color=color[2], linestyle='dashed',
                        linewidth=linewidth)

            axes2.tick_params(labelsize=font_size_labels)

            axes2.set_ylim(bottom=bottom, top=top)
            print(f"{model2, statistics.median(present_ppl), statistics.median(absent_ppl)}")

        model3 = 'one2seq_'
        scores, predictions, entropies, context_lines = json_load_one2seq(model3, dataset.lower())
        present_ppl, absent_ppl = get_one2seq_ppl(predictions, scores, context_lines)

        with sns.color_palette("Set2"):
            axes2 = plt.subplot(4, 3, j, sharey=axes1)

            if i == 0:
                axes2.set_title('CatSeq-Transformer', fontsize=font_size)
            j += 1

            axes2.set_xlabel(name, fontsize=font_size)
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

            # statistics.median(present_ppl)
            plt.axvline(statistics.median(present_ppl), color=color[1], linestyle='dashed', linewidth=linewidth)
            plt.axvline(statistics.median(absent_ppl), color=color[2], linestyle='dashed',
                        linewidth=linewidth)

            axes2.tick_params(labelsize=font_size_labels)

            axes2.set_ylim(bottom=bottom, top=top)
            print(f"{model2, statistics.median(present_ppl), statistics.median(absent_ppl)}")

        # was orig commented
        # present_ppl, absent_ppl= normalize(present_ppl, absent_ppl)



    fig.tight_layout(pad=1.5)

    plt.savefig('graphs/new_perplexities_' + model1 + '_' + model2 + '_'+model3+'new.png')
    plt.show()
    plt.close()

#plot_histogram_transformers()
'''



############################################################################################################################################################################################   New 

def process_predictions(predictions, beam=False):
    stemmer = PorterStemmer()

    processed_predictions = []
    for beam_prediction in predictions:
        if beam:
            prediction_ = ""
            for prediction in beam_prediction:
                prediction = prediction.replace(";", "<sep>")
                prediction = prediction.split("<eos>")[0]
                if not prediction_:
                    prediction_ += prediction
                else:
                    prediction_ += ' <sep> ' + prediction
            prediction = prediction_
        else:
            beam_prediction = beam_prediction.replace(";", "<sep>")
            prediction = beam_prediction.split("<eos>")[0]

        prediction = prediction.split(",")

        stemed_prediction = []
        for kp in prediction:
            kp = kp.lower().strip()
            if kp != "" and kp != "<peos>" and kp!="," and kp != "." and kp != "<unk>":  # and "." not in kp and "," not in kp
                tokenized_kp = kp.split(" ")  # nltk.word_tokenize(kp)
                tokenized_stemed_kp = [stemmer.stem(kw).strip() for kw in tokenized_kp]
                stemed_kp = " ".join(tokenized_stemed_kp).replace("< digit >", "<digit>")
                if stemed_kp.strip() != "":
                    stemed_prediction.append(stemed_kp.strip())

        # make prediction duplicates free but preserve order for @topk

        prediction_dict = {}
        stemed_prediction_ = []
        for kp in stemed_prediction:
            if kp not in prediction_dict:
                prediction_dict[kp] = 1
                stemed_prediction_.append(kp)
        stemed_prediction = stemed_prediction_

        processed_predictions.extend(stemed_prediction)

    return processed_predictions

def process_srcs(srcs):
    stemmer = PorterStemmer()
    processed_srcs = []
    tokenized_src = srcs.split()  # Split the string into words
    tokenized_stemed_src = [stemmer.stem(token.strip().lower()).strip() for token in tokenized_src]
    stemed_src = " ".join(tokenized_stemed_src).strip()
    processed_srcs.append(stemed_src)
    return processed_srcs

def add_sep_and_kpp(token_predictions, all_kpp_values):
    new_token_list = []
    new_kpp_list = []

    for tokens, kpp_values in zip(token_predictions, all_kpp_values):
        merged_tokens = []
        merged_kpp = []
        kpp_index = 0

        for i, sublist in enumerate(tokens):
            merged_tokens.extend(sublist)
            merged_kpp.extend(kpp_values[kpp_index:kpp_index + len(sublist)])
            kpp_index += len(sublist)

            if i < len(tokens) - 1:
                merged_tokens.append('<sep>')
                merged_kpp.append(0)  # Add corresponding 0 for <sep>

        new_token_list.append(merged_tokens)
        new_kpp_list.append(merged_kpp)

    return new_token_list, new_kpp_list

############################################################################################################################################################################## New
def plot_histogram_transformers_horizontal(datasets=['inspec']):
    def plot_distribution(ax, data, label, color, bins_num, min_lim, max_lim, linewidth, linestyle='dashed', alpha=0.5):
        # Plot histogram
        hist_plot = sns.histplot(data, bins=bins_num, color=color, label=label, kde=False, ax=ax,
                                binrange=(min_lim, max_lim), alpha=alpha)
        
        # Remove the black border around bins
        for patch in hist_plot.patches:
            patch.set_edgecolor('none')
        
        # Add vertical line for median
        ax.axvline(statistics.median(data), color=color, linestyle=linestyle, linewidth=linewidth)


    bins_num = 25
    min_lim = 1
    max_lim = 4.5
    linewidth = 1.5
    font_size_extra = 14
    font_size = 10
    color = sns.color_palette("bright")
    rotation_yticks = 0
    fig = plt.figure(figsize=[30, 10])
    
    fig1, axs = plt.subplots(7, len(datasets), squeeze=False, figsize=[8, 9])

    models = [
        ('exhird_h_', get_ppl, 'ExHiRD'),
        ('one2seq_', get_one2seq_ppl, 'Transformer'),
        ('one2set_', get_one2seq_ppl, 'Trans2set'),
        ('t5_', calculate_bart_ppl, 'T5'),
        ('bart_', calculate_bart_ppl, 'BART'),
        ('llama', get_one2seq_ppl, 'Llama 3'),
        ('phi', get_one2seq_ppl, 'Phi 3')
    ]

    for i, dataset in enumerate(datasets):
        for j, (model, ppl_func, ylabel) in enumerate(models):
            present_ppl, absent_ppl = [], []

            if model == 'exhird_h_':
                scores, predictions, _ = json_load(model, dataset.lower())
                present_ppl, absent_ppl = ppl_func(predictions, scores, model)
            elif model in ['t5_', 'bart_']:
                probs, tokens, src, _ = load_bart_tokens(dataset.lower(), model=model)
                present_ppl, absent_ppl = ppl_func(probs, tokens, src, model=model)
            elif model == 'llama':
                kp_predictions, probabilities, token_predictions, src, targets, all_kpp_values=load_llama(dataset) 
                context_lines = src
                token_predictions, scores = add_sep_and_kpp(token_predictions, probabilities)
                predictions = [process_predictions(tp) for tp in token_predictions]
                present_ppl, absent_ppl = ppl_func(predictions, scores, context_lines)
            elif model == 'phi':
                kp_predictions, probabilities, token_predictions, src, targets, all_kpp_values= load_phi(dataset)
                context_lines = src
                token_predictions, scores = add_sep_and_kpp(token_predictions, probabilities)
                predictions = [process_predictions(tp) for tp in token_predictions]
                present_ppl, absent_ppl = ppl_func(predictions, scores, context_lines)
            elif model == 'one2set_':
                scores, predictions, context_lines = json_load_one2set(model, dataset.lower())
                present_ppl, absent_ppl = get_one2seq_ppl(predictions, scores, context_lines)
            elif model == 'one2seq_':
                scores, predictions, context_lines = json_load_one2seq(model, dataset.lower())
                present_ppl, absent_ppl = get_one2seq_ppl(predictions, scores, context_lines)
            else:
                raise ValueError(f"Unknown model: {model}")

            # Plot present and absent perplexities
            plot_distribution(axs[j, i], present_ppl, "present", color[1], bins_num, min_lim, max_lim, linewidth)
            plot_distribution(axs[j, i], absent_ppl, "absent", color[2], bins_num, min_lim, max_lim, linewidth)

            # Set labels and tick parameters
            name = 'KP20k' if dataset == 'kp20k' else dataset.capitalize()
            axs[j, i].set_xlabel(name, fontsize=font_size)
            axs[j, i].xaxis.set_label_position('top')
            axs[j, i].tick_params(axis='y', labelrotation=rotation_yticks)
            axs[j, i].tick_params(axis='x', bottom=False, labelbottom=False)
            
            # Set x-axis labels only for the first row
            if j == 0:
                name = 'KP20k' if dataset == 'kp20k' else dataset.capitalize()
                axs[j, i].set_xlabel(name, fontsize=font_size)
                axs[j, i].xaxis.set_label_position('top')
            else:
                axs[j, i].set_xlabel('')
                axs[j, i].xaxis.set_label_position('top')
                
            # Set y-axis labels only for the last Column
            if i == len(datasets) - 1:
                axs[j, i].set_ylabel(ylabel, fontsize=font_size)
                axs[j, i].yaxis.set_label_position('right')
            else:
                axs[j, i].set_ylabel('')
                axs[j, i].yaxis.set_label_position('right')

            # Set consistent ylim
            bottom, top = axs[0, i].get_ylim()
            axs[j, i].set_ylim(bottom=bottom, top=top)
            if i == len(datasets) - 1 and j == 0:
                axs[j, i].legend(frameon=False, prop={'size': 7}, loc='upper right')

            print(f"{model, statistics.median(present_ppl), statistics.median(absent_ppl)}")

    fig1.tight_layout()
    plt.savefig(f'graphs/horizontal_perplexities_{datasets[0]}_new.png')
    plt.show()
    plt.close()

plot_histogram_transformers_horizontal(['kp20k','krapivin','inspec', 'semeval'])



def print_tokens(tokens, probs, old_t5=False):
    data = []
    concat = ''
    if old_t5:
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        concat = tokenizer.decode(tokens)
    for i, token in enumerate(tokens):
        if old_t5:
            token = tokenizer.decode(token)

        data.append({token:'{:.3f}'.format(probs[i])})

    return data, concat

def get_keyphrases(tokens, delimiter=';'):
    kps= [kp.strip() for kp in ''.join(tokens).split(delimiter)]
    kps[-1] = kps[-1][:-4]
    return kps
def get_present_absent(kps, stem_src):
    present, absent = [], []
    for kp in kps:
        if stem_text(kp) in stem_src:
            present.append(kp)
        else:
            absent.append(kp)
    return present, absent
def test_both(dataset='inspec'):
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_probs, t5_tokens, t5_src, t5_target = load_bart_tokens(dataset.lower(), model='t5_')
    bart_probs, bart_tokens, bart_src, bart_target = load_bart_tokens(dataset)

    ppl, kp_predictions, context_lines, probabilities, predicted_tokens = json_load_dump(dataset.lower(), probab=True)
    #print(t5_tokenizer.decode(predicted_tokens[1][0]))

    for i, src in enumerate(t5_src):
        stemmed_src = stem_text(src)

        t5_data, _ = print_tokens(t5_tokens[i], t5_probs[i])
        bart_data, _ = print_tokens(bart_tokens[i], bart_probs[i])
        bart_kps = get_keyphrases(bart_tokens[i])
        bart_present, bart_absent = get_present_absent(bart_kps, stemmed_src)

        old_t5_data, kps = print_tokens(predicted_tokens[i][0], probabilities[i], old_t5=True)
        old_t5_kps = [kp.strip() for kp in kps.replace('<pad>', '').replace('</s>', '').split('<extra_id_-1>')]

        old_t5_present, old_t5_absent = get_present_absent(old_t5_kps, stemmed_src)

        print('old t5 present', old_t5_present)
        print('old t5 absent', old_t5_absent)
        print('old t5:',old_t5_data)
        print('bart present', bart_present)
        print('bart absent', bart_absent)
        print('bart:',bart_data)
        #print('old t5:', old_t5_data)
        print()


#test_both()

def plot_bins(ax, p_bins, a_bins, color, title, show_ylabels, show_legend=True):
    dic = {
        'probability': [],
        'legend': [],
        'token_pos': []
    }
    
    for i, bin in enumerate(p_bins):
        for probability in bin:
            dic['probability'].append(probability)
            dic['legend'].append('present')
            dic['token_pos'].append(i + 1)

    for i, bin in enumerate(a_bins):
        for probability in bin:
            dic['probability'].append(probability)
            dic['legend'].append('absent')
            dic['token_pos'].append(i + 1)
    
    df = pd.DataFrame(data=dic)
    sns.boxplot(x="token_pos", y="probability", hue="legend", data=df, ax=ax, showfliers=False,
                palette=[color[1], color[2]]).set(xlabel=None, ylabel=None, title=title)
    
    if not show_ylabels:
        ax.set(yticklabels=[])
    
    if show_legend:
        # Position the legend at the bottom-right
        ax.legend(bbox_to_anchor=(8, -0.25), ncol=2, frameon=False)
    else:
        ax.legend([], [], frameon=False)
        
    ax.set_ylim(0, 1.1)

def make_sns_boxplot(exhird_bins, one2_seq_bins, one2set_bins, t5_bins, bart_bins, llama_bins, phi_bins, filename, xlabel, ylabel, title, model1='ExHiRD', model2='T5', model3='One2Seq'):
    sns.set_theme(style="whitegrid", font_scale=1.3)
    color = sns.color_palette("pastel")
    
    fig, axes = plt.subplots(1, 7, figsize=[20, 5])
    plt.subplots_adjust(bottom=0.24, left=0.06)
    fig.text(0.001, 0.5, ylabel, va='center', rotation='vertical', fontsize=16)
    fig.text(0.45, 0.01, xlabel, va='center', fontsize=16)
    
    # Plot each model's bins, showing legend only for the first plot
    plot_bins(axes[0], *exhird_bins, color, model1, show_ylabels=True, show_legend=True)
    plot_bins(axes[1], *one2_seq_bins, color, 'Transformer', show_ylabels=False, show_legend=False)
    plot_bins(axes[2], *one2set_bins, color, 'Trans2Set', show_ylabels=False, show_legend=False)
    plot_bins(axes[3], *t5_bins, color, 'T5', show_ylabels=False, show_legend=False)
    plot_bins(axes[4], *bart_bins, color, 'Bart', show_ylabels=False, show_legend=False)
    plot_bins(axes[5], *llama_bins, color, 'Llama 3', show_ylabels=False, show_legend=False)
    plot_bins(axes[6], *phi_bins, color, 'Phi 3', show_ylabels=False, show_legend=False)

    plt.tight_layout()
    plt.savefig('graphs/' + filename + '.png', bbox_inches='tight', pad_inches=0.15)
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


def probab_transformer_boxplots(dataset, model = 'one2seq_'):

    if model == 'one2seq_':
        scores, predictions, context_lines = json_load_one2seq(model, dataset.lower())
    else:
        scores, predictions, context_lines = json_load_one2set(model, dataset.lower())

    #present_ppl, absent_ppl = get_one2seq_ppl(predictions, scores, context_lines)

    p_bins = [[] for i in range(5)]
    a_bins = [[] for i in range(5)]


    for i, pred in enumerate(predictions):
        stemmed_context = stem_text(context_lines[i])
        kp_collect = []
        prob_collect = []
        kp_preds = set()
        for j, token in enumerate(pred):
            if token == "<sep>":
                if len(kp_collect) > 0:

                    stemmed_kp = stem_text(' '.join(kp_collect))

                    if stemmed_kp not in kp_preds:
                        kp_preds.add(stemmed_kp)
                    else:
                        kp_collect, prob_collect = [], []
                        continue

                    if stemmed_kp in stemmed_context:
                        for num, prob in enumerate(prob_collect[:5]):
                            p_bins[num].append(prob)
                    else:
                        for num, prob in enumerate(prob_collect[:5]):
                            a_bins[num].append(prob)

                    kp_collect, prob_collect = [], []
            else:
                if len(pred) != len(scores[i]):
                    #print(kp_collect, len(pred))
                    break
                kp_collect.append(token)
                prob_collect.append(scores[i][j])
        if len(kp_collect) > 0:



            stemmed_kp = stem_text(' '.join(kp_collect))
            if stemmed_kp in kp_preds:
                continue

            if stemmed_kp in stemmed_context:
                for num, prob in enumerate(prob_collect[:5]):
                    p_bins[num].append(prob)
            else:
                for num, prob in enumerate(prob_collect[:5]):
                    a_bins[num].append(prob)

    return [p_bins, a_bins]

def probab_llama_boxplots(dataset): #taken from line 811 one2seq code and adapted for llama

    kp_predictions, probabilities, token_predictions, src, targets, all_kpp_values=load_llama(dataset)
    context_lines=src
    token_predictions,scores=add_sep_and_kpp(token_predictions, probabilities)
    
    predictions=[]
    srcs=[]
    
    for i in range(len(src)):
        temp_predictions = process_predictions(token_predictions[i])
        #temp_srcs = process_srcs(src[i])
        predictions.append(temp_predictions)
        #srcs.append(temp_srcs)
        
        
    p_bins = [[] for i in range(5)]
    a_bins = [[] for i in range(5)]


    for i, pred in enumerate(predictions):
        stemmed_context = stem_text(context_lines[i])
        kp_collect = []
        prob_collect = []
        kp_preds = set()
        for j, token in enumerate(pred):
            if token == "<sep>":
                if len(kp_collect) > 0:

                    stemmed_kp = stem_text(' '.join(kp_collect))

                    if stemmed_kp not in kp_preds:
                        kp_preds.add(stemmed_kp)
                    else:
                        kp_collect, prob_collect = [], []
                        continue

                    if stemmed_kp in stemmed_context:
                        for num, prob in enumerate(prob_collect[:5]):
                            p_bins[num].append(prob)
                    else:
                        for num, prob in enumerate(prob_collect[:5]):
                            a_bins[num].append(prob)

                    kp_collect, prob_collect = [], []
            else:
                if len(pred) != len(scores[i]):
                    #print(kp_collect, len(pred))
                    break
                kp_collect.append(token)
                prob_collect.append(scores[i][j])
        if len(kp_collect) > 0:



            stemmed_kp = stem_text(' '.join(kp_collect))
            if stemmed_kp in kp_preds:
                continue

            if stemmed_kp in stemmed_context:
                for num, prob in enumerate(prob_collect[:5]):
                    p_bins[num].append(prob)
            else:
                for num, prob in enumerate(prob_collect[:5]):
                    a_bins[num].append(prob)
    return [p_bins, a_bins]
#probab_llama_boxplots("semeval")                    

def probab_phi_boxplots(dataset): #taken from line 811 one2seq code and adapted for llama

    kp_predictions, probabilities, token_predictions, src, targets, all_kpp_values=load_phi(dataset)
    context_lines=src
    token_predictions,scores=add_sep_and_kpp(token_predictions, probabilities)
    
    predictions=[]
    srcs=[]
    
    for i in range(len(src)):
        temp_predictions = process_predictions(token_predictions[i])
        #temp_srcs = process_srcs(src[i])
        predictions.append(temp_predictions)
        #srcs.append(temp_srcs)
        
        
    p_bins = [[] for i in range(5)]
    a_bins = [[] for i in range(5)]


    for i, pred in enumerate(predictions):
        stemmed_context = stem_text(context_lines[i])
        kp_collect = []
        prob_collect = []
        kp_preds = set()
        for j, token in enumerate(pred):
            if token == "<sep>":
                if len(kp_collect) > 0:

                    stemmed_kp = stem_text(' '.join(kp_collect))

                    if stemmed_kp not in kp_preds:
                        kp_preds.add(stemmed_kp)
                    else:
                        kp_collect, prob_collect = [], []
                        continue

                    if stemmed_kp in stemmed_context:
                        for num, prob in enumerate(prob_collect[:5]):
                            p_bins[num].append(prob)
                    else:
                        for num, prob in enumerate(prob_collect[:5]):
                            a_bins[num].append(prob)

                    kp_collect, prob_collect = [], []
            else:
                if len(pred) != len(scores[i]):
                    #print(kp_collect, len(pred))
                    break
                kp_collect.append(token)
                prob_collect.append(scores[i][j])
        if len(kp_collect) > 0:



            stemmed_kp = stem_text(' '.join(kp_collect))
            if stemmed_kp in kp_preds:
                continue

            if stemmed_kp in stemmed_context:
                for num, prob in enumerate(prob_collect[:5]):
                    p_bins[num].append(prob)
            else:
                for num, prob in enumerate(prob_collect[:5]):
                    a_bins[num].append(prob)
    return [p_bins, a_bins]


def box_plots(dataset):
    exhird_bins = probab_exhird_boxplots(dataset)
    t5_bins = probab_t5_boxplots(dataset)
    bart_bins = probab_bart_boxplots(dataset)
    transformer = probab_transformer_boxplots(dataset)
    trans2set = probab_transformer_boxplots(dataset, model = 'one2set_')
    llama_bins=probab_llama_boxplots(dataset)
    p_bins, a_bins=llama_bins
    print(p_bins)
    phi_bins=probab_phi_boxplots(dataset)


    make_sns_boxplot(exhird_bins, transformer, trans2set, bart_bins, t5_bins,llama_bins,phi_bins,
                     'all_5_boxplot_' + dataset + '_present_absent_', 'Token position',
                     'Probability', '')
#box_plots('semeval')

def line_plot_reliability(model,json_name, num_buckets, color, num, axes_name):
    with open('data_dump/'+model+'_'+json_name+'.json', 'r') as f:
        data = json.load(f)
        #print(data)
    for i, dataset in enumerate(list(data.keys())):
        bucket_values = data[dataset]

        # change to output/calibration_train_test_ for temperature scaled reliability diagram

        X = [1/(num_buckets) * i + (1/num_buckets)/2 for i in range(num_buckets) if bucket_values[i] > 0]
        bucket_accuracy = [i for i in bucket_values if i > 0]

        if dataset == 'semeval':
            dataset = 'SemEval'
        elif dataset == 'kp20k':
            dataset = 'KP20k'
        else:
            dataset = dataset.capitalize()
        if i==len(list(data.keys())) - 1:
            num+=1

        axes_name = sns.lineplot(x=X, y=bucket_accuracy, color=color[i *2 +num],
                             label=dataset.capitalize(),  marker='s')
        axes_name = sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--', color='black')


def plot_reliability(json_name, plot_name, num_buckets=10, model1 = 'exhird', model2 ='one2seq', model3='one2set', model4='t5', model5='bart',model6='llama',model7='phi'):

    font_size = 16
    font_size_labels = 11
    color = sns.color_palette("bright")
    sns.set_theme(style='whitegrid', font_scale=1.3,  rc = {'axes.facecolor': '#FFEED2'})
    fig = plt.figure(figsize=[22,5])
    fig.text(0.001, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=font_size)
    fig.text(0.45, 0.015, 'Confidence', va='center', fontsize=font_size)
    #fig.text(0.4, 0.98, plot_name, va='center', fontsize=font_size + 2)

    axes1 = plt.subplot(1, 7, 1)

    num=2   #factor to change color
    line_plot_reliability(model1, json_name, num_buckets, color, num, axes1)

    axes2 = plt.subplot(1, 7, 2)
    line_plot_reliability(model2, json_name, num_buckets, color, num, axes2)

    axes3 = plt.subplot(1, 7, 3)
    line_plot_reliability(model3, json_name, num_buckets, color, num, axes3)

    axes4 = plt.subplot(1, 7, 4)
    line_plot_reliability(model4, json_name, num_buckets, color, num, axes4)
    
    axes5 = plt.subplot(1, 7, 5)
    line_plot_reliability(model5, json_name, num_buckets, color, num, axes5)
    
    axes6 = plt.subplot(1, 7, 6)
    line_plot_reliability(model6, json_name, num_buckets, color, num, axes5)
    
    axes7 = plt.subplot(1, 7, 7)
    line_plot_reliability(model7, json_name, num_buckets, color, num, axes5)
    
    
    fig.tight_layout(pad=1.15)
    axes1.set_title("ExHiRD", fontsize=font_size)
    axes2.set_title("Transformer", fontsize=font_size)
    axes3.set_title("Trans2Set", fontsize=font_size)
    axes4.set_title("T5", fontsize=font_size)
    axes5.set_title("Bart", fontsize=font_size)
    axes6.set_title("Llama 3", fontsize=font_size)
    axes7.set_title("Phi 3", fontsize=font_size)
    
    axes1.set_xticks([1.0/(num_buckets) * i *2 for i in range(num_buckets//2 + 1)])
    axes2.set_xticks([1.0 / (num_buckets) * i * 2 for i in range(num_buckets//2 + 1)])
    axes3.set_xticks([1.0 / (num_buckets) * i * 2 for i in range(num_buckets // 2 + 1)])
    axes4.set_xticks([1.0 / (num_buckets) * i * 2 for i in range(num_buckets // 2 + 1)])
    axes5.set_xticks([1.0 / (num_buckets) * i * 2 for i in range(num_buckets // 2 + 1)])
    axes6.set_xticks([1.0 / (num_buckets) * i * 2 for i in range(num_buckets // 2 + 1)])
    axes7.set_xticks([1.0 / (num_buckets) * i * 2 for i in range(num_buckets // 2 + 1)])
    
    
    axes1.legend(frameon=False, prop={'size': 14})
    axes2.get_legend().remove()
    axes3.get_legend().remove()
    axes4.get_legend().remove()
    axes5.get_legend().remove()
    axes6.get_legend().remove()
    axes7.get_legend().remove()
    
    plt.ylim(top=1.05)
    plt.savefig('graphs/'+plot_name+'_'+model1+'_'+model2+'_'+model3+'.png')
    plt.show()
    plt.close()

#plot_reliability('calibrate_kpp_values', plot_name='Calibration_5_models', num_buckets=10)

def plot_relative_pos_graph(model1 = 'exhird_h_', model2='t5', model3='one2seq_', model4='bart', model5 = 'one2set_', model6='llama',model7='phi'):
    datasets =[ 'kp20k','krapivin', 'inspec','semeval']
    #datasets=['semeval','krapivin']
    percentages = ['0-20', '20-40', '40-60', '60-80', '80-100']


    font_size = 10
    font_size_labels = 11
    color = sns.color_palette("pastel")
    j = 1
    fig = plt.figure(figsize=[10, 6])
    fig.text(0.001, 0.5, 'Accuracy ', va='center', rotation='vertical', fontsize=font_size)

    fig.text(0.46, 0.025, 'Positional range', va='center', fontsize=font_size)
            
    X = np.array([0.10*i for i in range(5)])
    for i, dataset in enumerate(datasets):
        print(dataset)
        #exhird_errors, t5_errors = get_relative_error_numbers(dataset)
        exhird_errors = [(1-error) for error in  get_relative_error_numbers(model1, dataset.lower())]
        t5_errors = [(1-error) for error in  get_relative_error_numbers(model2, dataset.lower())]
        one2_errors= [(1-error) for error in  get_relative_error_numbers(model3, dataset.lower())]
        bart_errors = [(1-error) for error in  get_relative_error_numbers(model4, dataset.lower())]
        trans2set_errors = [(1-error) for error in  get_relative_error_numbers(model5, dataset.lower())]
        llama_errors=   [(1-error) for error in  get_relative_error_numbers(model6, dataset.lower())] 
        phi_errors=   [(1-error) for error in  get_relative_error_numbers(model7, dataset.lower())]    
        axes1 = plt.subplot(2, 2, j)



        #ax = fig.add_axes([0, 0, 1, 1])
        width = 0.012
        j += 1
        
        axes1.bar(X, exhird_errors, color=color[1], width=width, label='ExHiRD', edgecolor='black')
        axes1.bar(X + width, t5_errors, color=color[2], width=width, label='T5', edgecolor='black')
        axes1.bar(X + 2*width, one2_errors, color=color[4], width=width, label='Transformer', edgecolor='black')
        axes1.bar(X + 3*width, bart_errors, color=color[8], width=width, label='Bart', edgecolor='black')
        axes1.bar(X + 4*width, trans2set_errors, color=color[3], width=width, label='Trans2set', edgecolor='black')
        axes1.bar(X + 5*width, llama_errors, color=color[5], width=width, label='Llama3', edgecolor='black')
        axes1.bar(X + 6*width, phi_errors, color=color[6], width=width, label='Phi3', edgecolor='black')
        
        if dataset[0].islower():
            name = dataset.capitalize()
        else:
            name = dataset
        axes1.set_xlabel(name, fontsize=font_size)
        axes1.autoscale(enable=True, axis='x', tight=True)
        plt.ylim(bottom=0, top=0.65)
        plt.xticks(X + 6*width / 2, tuple(percentages))
        plt.grid(axis='y', color='gray', linestyle = 'dashed', alpha=0.4)
        plt.grid(axis='y', color='gray', linestyle = 'dashed', alpha=0.4)
        plt.grid(axis='y', color='gray', linestyle = 'dashed', alpha=0.4)

        #if i == 0:
            #axes1.legend(frameon=False, prop={'size': 8}, ncol=3)


        axes1.tick_params(labelsize=font_size_labels)
        #print(i)
        #plt.setp(axes1.get_xticklabels(), visible=True)
    labels = ['ExHiRD', 'T5', 'Transformer', 'Bart', 'Trans2set','Llama3','Phi3']
    fig.legend(labels=labels, loc="upper center", ncol=4)


    fig.tight_layout(pad=1.5)

    plt.savefig('graphs/relative_pos_comparison_all_datasets_'+model1+'_'+model2+'_'+model3+model4+model5+'.png')
    plt.show()
    plt.close()


#plot_relative_pos_graph()