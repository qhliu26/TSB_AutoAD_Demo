from constant import *
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import operator
import math
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import networkx
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
from collections import Counter
import pickle
import pycatch22 as catch22
from pathlib import Path

def plot_box_plot(all_df, target_df, methods_variant, metric_name):
    if len(target_df.columns) > 0:

        eval_list = []
        for index, row in all_df.iterrows():
            for method in methods_variant:
                eval_list.append([method, row['file'], row[method]])
        eval_df = pd.DataFrame(eval_list, columns=['classifier_name', 'dataset_name', 'accuracy'])
        p_values, average_ranks, _ = Friedman_Nemenyi(df_perf=eval_df, alpha=0.05)
        order = average_ranks.keys().to_list()[::-1]

        fig = plt.figure(figsize=(10, min(30, max(1, int(0.40*len(target_df.columns))))))
        g = sns.boxplot(data=target_df, order=order, palette="husl", showfliers = False, orient='h')
        # plt.title("Figure 1: Boxplot", fontsize=18)
        st.markdown("<h3 style='text-align: center;'>Figure 1: Boxplot</h3>", unsafe_allow_html=True)
        plt.xlabel(metric_name, fontsize=16)
        st.pyplot(fig)

        st.markdown('<hr style="border:2px solid gray">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Figure 2: Critical Diagram (α=0.05)</h3>", unsafe_allow_html=True)

        graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                    cd=None, reverse=True, width=int(len(methods_variant)*0.8), textspace=2.5)


def get_bubble_data(acc_df, det_time_df, exe_time_df, methods_variant):
    method_to_display_det = []
    for m in methods_variant:
        if m in det_time_df['Method'].unique():
            method_to_display_det.append(m)
    method_to_display_exe = []
    for m in methods_variant:
        if m in exe_time_df['Method'].unique():
            method_to_display_exe.append(m)
    
    det_time = []
    for m in method_to_display_det:
        det_time.append(det_time_df[det_time_df['Method']==m]['Time'].values.tolist()[0])
    det_time_text = []
    for t in det_time:
        det_time_text.append('Detection Time: ' + str(t))

    exe_time = []
    for m in method_to_display_exe:
        exe_time.append(exe_time_df[exe_time_df['Method']==m]['Time'].values.tolist()[0])
    exe_time_text = []
    for t in exe_time:
        exe_time_text.append('Execution Time: ' + str(t))

    acc_det = []
    for m in method_to_display_det:
        acc_det.append(acc_df[m].mean())
    acc_exe = []
    for m in method_to_display_exe:
        acc_exe.append(acc_df[m].mean())

    det_idxs = np.argsort(det_time) 
    det_time = np.array(det_time)[det_idxs]
    det_time_text = np.array(det_time_text)[det_idxs]
    methods_det = np.array(method_to_display_det)[det_idxs]
    acc_det = np.array(acc_det)[det_idxs]

    exe_idxs = np.argsort(exe_time) 
    exe_time = np.array(exe_time)[exe_idxs]
    exe_time_text = np.array(exe_time_text)[exe_idxs]
    methods_exe = np.array(method_to_display_exe)[exe_idxs]
    acc_exe = np.array(acc_exe)[exe_idxs]

    return det_time, det_time_text, exe_time, exe_time_text, methods_det, methods_exe, acc_det, acc_exe

def Friedman_Nemenyi(alpha=0.05, df_perf=None):
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # Record the maximum number of datasets
    max_nb_datasets = df_counts['count'].max()
    # Create a list of classifiers
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])

    # print('classifiers: ', classifiers)

    '''
    Expected input format for friedmanchisquare is:
                Dataset1        Dataset2        Dataset3        Dataset4        Dataset5
    classifer1
    classifer2
    classifer3 
    '''

    # Compute friedman p-value
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]

    # Decide whether to reject the null hypothesis
    # If p-value >= alpha: we cannot reject the null hypothesis. No statistical difference.
    if friedman_p_value >= alpha:
        return None,None,None
    # Friedman test OK
    # Prepare input for Nemenyi test
    data = []
    for c in classifiers:
        data.append(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
    data = np.array(data, dtype=np.float64)
    # Conduct the Nemenyi post-hoc test
    # print(classifiers)
    # Order is classifiers' order
    nemenyi = posthoc_nemenyi_friedman(data.T)

    # print(nemenyi)
    
    # Original code: p_values.append((classifier_1, classifier_2, p_value, False)), True: represents there exists statistical difference
    p_values = []

    # Comparing p-values with the alpha value
    for nemenyi_indx in nemenyi.index:
        for nemenyi_columns in nemenyi.columns:
            if nemenyi_indx < nemenyi_columns:
                if nemenyi.loc[nemenyi_indx, nemenyi_columns] < alpha:
                    p_values.append((classifiers[nemenyi_indx], classifiers[nemenyi_columns], nemenyi.loc[nemenyi_indx, nemenyi_columns], True))
                else:
                    p_values.append((classifiers[nemenyi_indx], classifiers[nemenyi_columns], nemenyi.loc[nemenyi_indx, nemenyi_columns], False))
            else: continue

    # Nemenyi test OK

    m = len(classifiers)

    # Sort by classifier name then by dataset name
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])

    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)

    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=np.unique(sorted_df_perf['dataset_name']))

    dfff = df_ranks.rank(ascending=False)
    # compute average rank
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    
    return p_values, average_ranks, max_nb_datasets

def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=200, textspace=1, reverse=False, filename=None, **kwargs):
    
    width = width
    textspace = float(textspace)
    '''l is an array of array 
        [[......]
         [......]
         [......]]; 
    n is an integer'''
    # n th column
    def nth(l, n):
        n = lloc(l, n)
        # Return n th column
        return [a[n] for a in l]
    
    '''l is an array of array 
        [[......]
         [......]
         [......]]; 
    n is an integer'''
    # return an integer, count from front or from back.
    def lloc(l, n):
        if n < 0:
            return len(l[0]) + n
        else:
            return n
    # lr is an array of integers
    # Maximum range start from all zeros. Returns an iterable element of tuple.
    def mxrange(lr):
        # If nothing in the array
        if not len(lr):
            yield ()
        else:
            index = lr[0]
            # Check whether index is an integer.
            if isinstance(index, int):
                index = [index]
            # *index: index must be an iterable []
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    # Form a tuple, and generate an iterable value
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums
    # lowv: low value
    if lowv is None:
        '''int(math.floor(min(ssums))): select the minimum value in ssums and take floor.
           Then compare with 1 to see which one is the minimum.'''
        lowv = min(1, int(math.floor(min(ssums))))
    # highv: high value
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4
    # how many algorithms
    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace
    
    # Position of rank
    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        # Set up the format
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # set up the formats
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant + 2

    # matplotlib figure format setup
    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    hf = 1. / height
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    # Line plots
    def line(l, color='k', **kwargs):
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    # Add text to the plot
    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None

    # [lowv, highv], step size is 0.5
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        # If a is an integer
        if a == int(a):
            tick = bigtick
        # Plot a line
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    # Add text to the plot, only for integer value
    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    # Format for the first half of algorithms
    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)
        if nnames[i] in ['Oracle', 'GB', 'SS', 'Random (TS)', 'Random (D)']:
            color = 'b'
        else:
            color = 'k'
        text(textspace - 0.2, chei, filter_names(nnames[i]), color=color, ha="right", va="center", size=16)


    # Format for the second half of algorithms
    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)
        if nnames[i] in ['Oracle', 'GB', 'SS', 'Random (TS)', 'Random (D)']:
            color = 'b'
        else:
            color = 'k'
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]), color=color, ha="left", va="center", size=16)
        
    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=linewidth_sign)
            start += height
            
    start = cline + 0.2
    side = -0.02
    height = 0.1


    #Generate cliques and plot a line to connect elements in cliques    
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    # Plot a line to connect elements in cliques
    for clq in cliques:
        if len(clq) == 1:
            continue
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        # Test
        # print("ssums[min_idx]: {}; ssums[max_idx]: {}".format(ssums[min_idx], ssums[max_idx]))
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
        start += height
    # plt.title("Figure 2: Critical Diagram ({}=0.05)".format(r'$\alpha$'), fontsize=18)
    st.pyplot(fig)

def form_cliques(p_values, nnames):
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1
    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)

def add_rect(label, data):
	anom_plt = [None]*len(data)
	ts_plt = data.copy()
	len_ts = len(data)
	for i, lab in enumerate(label):
		if lab == 1:
			anom_plt[i] = data[i]
			anom_plt[min(len_ts-1, i+1)] = data[min(len_ts-1, i+1)]
	return anom_plt

def split_ts(data, window_size):
    # Compute the modulo
    modulo = data.shape[0] % window_size

    # Compute the number of windows
    k = data[modulo:].shape[0] / window_size
    assert(math.ceil(k) == k)

    # Split the timeserie
    data_split = np.split(data[modulo:], k)
    if modulo != 0:
        data_split.insert(0, list(data[:window_size]))
    data_split = np.asarray(data_split)

    return data_split

def run_model(ts_data, method_selected_exp):
    
    if method_selected_exp == 'UReg':
        ts_win = split_ts(ts_data, window_size=1024)
        meta_mat = np.zeros([ts_win.shape[0], 24])
        for i in range(ts_win.shape[0]):
            catch24_output = catch22.catch22_all(list(ts_win[i].ravel()), catch24=True)
            meta_mat[i, :] = catch24_output['values']
        meta_mat = pd.DataFrame(meta_mat)
        meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)
        result_path = f'data/model/UReg.pkl'
        with open(result_path, 'rb') as file:
            result = pickle.load(file)

        U_pred = np.column_stack([rf.predict(meta_mat) for rf in result['models']])
        prediction_scores = U_pred.dot(result['DVt'])
        preds = np.argmax(prediction_scores, axis=1)
        counter = Counter(preds)
        most_voted = counter.most_common(1)
        det = Candidate_Model_Set[int(most_voted[0][0])]
        success = True
    elif method_selected_exp == 'CLF':
        ts_win = split_ts(ts_data, window_size=1024)
        meta_mat = np.zeros([ts_win.shape[0], 24])
        for i in range(ts_win.shape[0]):
            catch24_output = catch22.catch22_all(list(ts_win[i].ravel()), catch24=True)
            meta_mat[i, :] = catch24_output['values']
        meta_mat = pd.DataFrame(meta_mat)
        meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)
        model_path = 'data/model/CLF.pkl'        
        filename = Path(model_path)
        with open(f'{filename}', 'rb') as input:
            model = pickle.load(input)
        preds = model.predict(meta_mat)
        counter = Counter(preds)
        most_voted = counter.most_common(1)
        det = Candidate_Model_Set[int(most_voted[0][0])]
        success = True
    else:
        st.markdown("Method not supported yet... Please visit our Github repo for more information (https://github.com/TheDatumOrg/AutoTSAD)")
        success = False
    
    if success:
        return det, success
    else:
        return Candidate_Model_Set, success
    
