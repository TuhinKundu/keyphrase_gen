import json
import seaborn as sns
import statistics
import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib.axes._axes import Axes
from utilities import *
import jsonlines

from utilities.post_process_utils import *

def normalize(present, absent, cnt=200):
    if len(present ) + len(absent) > cnt:
        rate = len(present) + len(absent)
        ratio = cnt / rate
        random.shuffle(present)
        present = present[: int(len(present) * ratio)]
        random.shuffle(absent)
        absent = absent[: int(len(absent) * ratio)]
    else:
        print('Data not enough!')
        assert 1==2
    return present, absent

def json_load(model, dataset):
    with open('graph_outputs/'+model+dataset+'_all_output.json', 'r') as f:
        dic= json.load(f)

    scores = dic['scores']
    predictions = dic['predictions']
    entropies = dic['entropies']

    return scores, predictions, entropies




def get_ppl(predictions, scores, model):
    present_ppl=[]
    absent_ppl = []

    for i, pred in enumerate(predictions):
        if 'catseq' in model:
            pred= pred[0].split(' ')

        kp_type = ''
        kp_probabilties = []
        for j, token in enumerate(pred):

            if token == '<p_start>':
                kp_type = '<p_start>'
            elif token == '<a_start>':
                kp_type = '<a_start>'
            elif token ==';':
                ppl = np.prod(kp_probabilties)**(-1/float(len(kp_probabilties)))
                if kp_type == '<p_start>':
                    present_ppl.append(ppl)
                else:
                    absent_ppl.append(ppl)
            else:
                kp_probabilties.append(scores[i][j])

    return present_ppl, absent_ppl

def plot_histogram(dataset):
    model = 'exhird_h_'
    scores, predictions, entropies = json_load(model, dataset)
    present_ppl, absent_ppl = get_ppl(predictions, scores, model)
    #present_ppl, absent_ppl= normalize(present_ppl, absent_ppl)

    bins_num=28
    min_lim = 1
    max_lim = 7
    linewidth = 1.5
    font_size=12
    color = sns.color_palette("coolwarm", 7)

    fig = plt.figure()
    fig.text(0.005, 0.5, 'Count', va='center', rotation='vertical', fontsize=font_size)
    axes1 = plt.subplot(211)
    axes1 = sns.distplot(present_ppl, bins=bins_num,
                         hist_kws={'range': [min_lim, max_lim]},
                         hist=True,
                         kde=False,
                         color=color[0],
                         label="Present kp"
                         )
    axes1.set_title(dataset)
    axes1.set_ylabel(model[:-1])
    #axes1.set_xlabel('Perplexities')
    axes1 = sns.distplot(absent_ppl, bins=bins_num, hist_kws={'range': [min_lim, max_lim]},
                         hist=True,
                         kde=False,
                         color=color[-1],
                         label="Absent kp"
                         )
    plt.axvline(statistics.median(present_ppl), color=color[0], linestyle='dashed', linewidth=linewidth)
    plt.axvline(statistics.median(absent_ppl), color=color[-1], linestyle='dashed', linewidth=linewidth)

    axes1.legend(frameon=False)
    print(f"{statistics.median(present_ppl), statistics.median(absent_ppl),}")

    plt.setp(axes1.get_xticklabels(), visible=True)

    model = 'catseq_'
    scores, predictions, entropies = json_load(model, dataset)
    present_ppl, absent_ppl = get_ppl(predictions, scores, model)
    #present_ppl, absent_ppl= normalize(present_ppl, absent_ppl)

    with sns.color_palette("Set2"):
        axes2 = plt.subplot(212, sharex=axes1)
        # axes2.set_title("XSum",loc='left')
        axes2.set_ylabel(model[:-1])
        data = model
        axes2 = sns.distplot(present_ppl, bins=bins_num,
                             hist_kws={'range': [min_lim, max_lim], }, hist=True, kde=False,
                             # label=f"{ExistingBigram}",
                             color=color[0]
                             )
        axes2 = sns.distplot(absent_ppl, bins=bins_num,
                             hist_kws={'range': [min_lim, max_lim], }, hist=True, kde=False,
                             # label=f"{NovelBigram}",
                             color=color[-1]
                             )

        plt.axvline(statistics.median(present_ppl), color=color[0], linestyle='dashed', linewidth=linewidth)
        plt.axvline(statistics.median(absent_ppl), color=color[-1], linestyle='dashed',
                    linewidth=linewidth)
        # axes.legend(prop={'size': 10})
        # axes2.legend()
        # axes = sns.distplot(not_bigram_entropies,rug=True)
        # axes2.set_title('XSum')
        axes2.set_xlabel('Perplexities')

    fig.tight_layout()
    plt.savefig('graphs/'+dataset+'_perplexities.png')
    plt.show()



#plot_histogram(dataset)

def get_relative_ppl(predictions, scores, context_lines, model):
    relative_ppl = [[] for i in range(5)]
    bins = [0.2, 0.4, 0.6, 0.8, 1.0]

    for i, pred in enumerate(predictions):

        context = ' '.join([stemmer.stem(w) for w in context_lines[i].strip().split()])

        if 'catseq' in model:
            pred = pred[0].split(' ')

        kp_probabilties = []
        kp = []
        stemmed_pred = []
        pred_ppl = []
        for j, token in enumerate(pred):
            if token == ';':
                ppl = np.prod(kp_probabilties) ** (-1 / float(len(kp_probabilties)))
                pred_ppl.append(ppl)
                stemmed_pred.append(' '.join(kp))

                try:

                    pos = context.index(' '.join(kp[1:]))
                    kp = []
                    relative_pos = pos / float(len(context))
                    for k, bin in enumerate(bins):

                        if relative_pos<bin:
                            ind= k
                            break
                    relative_ppl[ind].append(ppl)

                except:
                    kp = []
                    continue


            else:
                kp_probabilties.append(scores[i][j])
                kp.append(stemmer.stem(token))
    return relative_ppl


def plot_sentence_pos():
    dataset = 'inspec'
    src = 'data/test_datasets/processed_'+dataset+'_testing_context.txt'
    context_file = open(src, encoding='utf-8')
    context_lines = context_file.readlines()

    #preds_file = open(opt.output, encoding='utf-8')
    #preds_lines = preds_file.readlines()

    model = 'exhird_h_'

    scores, predictions, entropies = json_load(model, dataset)
    relative_ppl1 = get_relative_ppl(predictions, scores, context_lines, model)

    make_boxplot(relative_ppl1, model + dataset, 'Relative pos', 'Perplexity', model + dataset)

    model = 'catseq_'
    scores, predictions, entropies = json_load(model, dataset)
    relative_ppl2 = get_relative_ppl(predictions, scores, context_lines, model)
    make_boxplot(relative_ppl2, model + dataset, 'Relative pos', 'Perplexity', model + dataset)

    keys = ['Relative Position', 'Perplexity']

    # axes = fig.add_axes([0.15, 0.3, 0.84, 0.66])
    # sns.distplot(x=ykey, data=df, hist=False, rug=True)
    # axes = sns.kdeplot(bigram_entropies)
    # axes = sns.kdeplot(not_bigram_entropies)
    #

    font_size =12
    fig = plt.figure()
    fig.text(1, 0.005, 'Count', va='center', rotation='horizontal', fontsize=font_size)
    colorblind = sns.color_palette("coolwarm", 10)[::-1]
    axes1: Axes = plt.subplot(121)
    max_lim = 7
    sns.boxplot(x=keys[0], y=keys[1], data=relative_ppl1,
                fliersize=0,
                # palette='coolwarm',
                # color=colorblind[3],
                palette=colorblind,
                # notch=True,
                )
    # axes1.tick_params(which='major', length=5)
    #axes1.set_xticks([0, 2, 4, 6, 8])
    axes1.set_xticklabels([0.2, 0.4, 0.6, 0.8, 1.0])

    # for box in axes1['boxes']:
    #     # change outline color
    #     # box.set(color='#7570b3', linewidth=2)
    #     # change fill color
    #     box.set(edgecolor='white')
    axes1.set_title(model[:-1])
    axes1.set_ylim(0, max_lim)
    # axes1.set_ylabel('')
    # axes1.legend()

    axes2 = plt.subplot(122, sharey=axes1)

    sns.boxplot(x=keys[0], y=keys[1], data=relative_ppl2,
                # notch=True,
                fliersize=0,
                palette=colorblind,
                # palette='Set2',
                # color=colorblind,
                )

    #axes2.set_xticks([0, 2, 4, 6, 8])
    axes2.set_xticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
    axes2.set_ylabel('')
    axes2.set_title(model[:-1])
    axes2.set_ylim(0, max_lim)
    fig.tight_layout()
    plt.savefig('graphs/' + dataset + '_relative_pos_ppl.png')
    plt.show()

#plot_sentence_pos()










