from nltk.stem.porter import *
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
stemmer = PorterStemmer()
def stem_text(text):

    return ' '.join([stemmer.stem(w) for w in text.strip().split()])

def make_consistent(kp1, kp2):
    kp1_split = kp1.split()
    kp2_split = kp2.split()

    if len(kp2_split) == len(kp1_split):
        return kp1_split, kp2_split
    elif len(kp1_split) > len(kp2_split):
        for i in range(len(kp1_split)-len(kp2_split)):
            kp2_split.append("<dummy>")
    else:
        for i in range(len(kp2_split)-len(kp1_split)):
            kp1_split.append("dummy")
    return kp1_split, kp2_split

def segregate_kps_for_one2set(tokens, context):
    stemmed_context = stem_text(context)
    present, absent = [], []
    kp_collect= []
    separator_token = '<sep>'
    for i, token in enumerate(tokens):
        if token == separator_token and len(kp_collect) >0:

            kp = ' '.join(kp_collect)
            kp = stem_text(kp)
            if kp in stemmed_context:
                present.append(kp)
            else:
                absent.append(kp)
            kp_collect = []
        else:
            kp_collect.append(token)

    return present, absent

def segregate_kps(kps, context, do_kp_stemming=True):
    """
    segregate keyphrases into present and absent
    append <dummy> keyphrase in case no keyphrase collected
    """

    present_kps, absent_kps = [], []
    for i, kp in enumerate(kps):
        if do_kp_stemming:
            stemmed_kp = stem_text(kp)
        else:
            stemmed_kp = kp
        if stemmed_kp in stem_text(context):
            present_kps.append(stemmed_kp)
        else:
            absent_kps.append(stemmed_kp)
    if len(present_kps) ==0:
        present_kps.append("<dummy>")
    if len(absent_kps) == 0:
        absent_kps.append("<dummy>")

    return present_kps, absent_kps

def remove_duplicates(predictions, ppl = None):

    score_pred = []
    set_pred = []

    for j, kp in enumerate(predictions):
        if kp not in set_pred:
            if ppl is not None:
                score_pred.append(ppl[j])
            set_pred.append(predictions[j])


    if ppl is not None:
        return set_pred, score_pred
    else:
        return set_pred

def calculate_error(n_samples, bucket_values, bucket_confidence, bucket_accuracy):
    """
    Computes several metrics used to measure calibration error:
        - Expected Calibration Error (ECE): \sum_k (b_k / n) |acc(k) - conf(k)|
        - Maximum Calibration Error (MCE): max_k |acc(k) - conf(k)|
        - Total Calibration Error (TCE): \sum_k |acc(k) - conf(k)|
    """

    assert len(bucket_values) == len(bucket_confidence) == len(bucket_accuracy)
    #assert sum(map(len, bucket_values)) == n_samples

    expected_error, max_error, total_error = 0., 0., 0.
    for (bucket, accuracy, confidence) in zip(
        bucket_values, bucket_accuracy, bucket_confidence
    ):

        if bucket > 0:
            delta = abs(accuracy - confidence)
            expected_error += (bucket / n_samples) * delta
            max_error = max(max_error, delta)
            total_error += delta
    return (expected_error * 100., max_error * 100., total_error * 100.)


def plot_reliability(json_name, plot_name, num_buckets=10, model1 = 'exhird', model2 ='t5'):

    font_size = 16
    font_size_labels = 11
    color = sns.color_palette("bright")
    sns.set_theme(style="darkgrid", font_scale=1.3)
    fig = plt.figure(figsize=[9,5])
    fig.text(0.001, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=font_size)
    fig.text(0.45, 0.015, 'Confidence', va='center', fontsize=font_size)
    #fig.text(0.4, 0.98, plot_name, va='center', fontsize=font_size + 2)

    axes1 = plt.subplot(1, 2, 1)
    with open('data_dump/'+model1+'_'+json_name+'.json', 'r') as f:
        data = json.load(f)

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

        axes1 = sns.lineplot(X, bucket_accuracy, color=color[i *2 +2],
                             label=dataset.capitalize(),  marker='s')
        axes1 = sns.lineplot([0, 1], [0, 1], linestyle='--', color='black')


    axes2 = plt.subplot(1, 2, 2)
    with open('data_dump/'+model2+'_' + json_name + '.json', 'r') as f:
        data = json.load(f)

    for i, dataset in enumerate(list(data.keys())):
        bucket_values = data[dataset]

        # change to output/calibration_train_test_ for temperature scaled reliability diagram

        X = [1 / (num_buckets) * i + (1 / num_buckets) / 2 for i in range(num_buckets) if bucket_values[i] > 0]
        bucket_accuracy = [i for i in bucket_values if i > 0]
        if dataset == 'semeval':
            dataset = 'SemEval'
        elif dataset == 'kp20k':
            dataset = 'KP20k'
        else:
            dataset = dataset.capitalize()
        axes2 = sns.lineplot(X, bucket_accuracy, color=color[i * 2 + 2],
                             label=dataset,  marker='s')
        axes2 = sns.lineplot([0, 1], [0, 1], linestyle='--', color='black')


    fig.tight_layout(pad=1.15)
    axes1.set_title("ExHiRD", fontsize=font_size)
    axes2.set_title("T5", fontsize=font_size)
    axes1.set_xticks([1.0/(num_buckets) * i *2 for i in range(num_buckets//2 + 1)])
    axes2.set_xticks([1.0 / (num_buckets) * i * 2 for i in range(num_buckets//2 + 1)])
    axes1.legend(frameon=False, prop={'size': 14})
    axes2.get_legend().remove()
    plt.ylim(top=1.05)
    plt.savefig('graphs/'+plot_name+'_'+model1+'_'+model2+'.png')
    plt.show()
    plt.close()


def partial_match_vector(predictions, targets, matrix, threshold):
    # print(predictions)
    seen = []
    markup_vector = [0 for _ in range(len(targets))]
    for i, pred in enumerate(predictions):
        max_value_pos = np.argmax(matrix[i])
        if matrix[i][max_value_pos] > threshold and targets[max_value_pos] not in seen:
            seen.append(targets[max_value_pos])
            markup_vector[max_value_pos] = 1

    # print([(targets[i], markup_vector[i]) for i in range(len(targets))])

    return markup_vector


def get_bucket(confidence, num_buckets):
    for i in range(num_buckets):
        if confidence < float((i + 1) / num_buckets):
            return i

