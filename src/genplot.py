import collections
import numpy as np
import matplotlib.pyplot as plt
from math import log

colors = {'DeepWalk': 'gold', 'SDNE': 'lightskyblue', 'FANE-weighted': 'purple', \
 'LINE(1st)': 'chocolate', 'LINE(2nd)': 'burlywood', 'LINE(1st+2nd)': 'wheat', \
 'Node2Vec': 'yellowgreen', 'FANE-sym':'magenta', 'FANE-row':'hotpink', 'NetMF': 'c'}  # change this when re-run

markers = {'DeepWalk': 'x', 'SDNE': 's', 'FANE-weighted': '.', \
 'LINE(1st)': 'v', 'LINE(2nd)': '^', 'LINE(1st+2nd)': '<', 'Node2Vec': '*', 'FANE-sym':'H', \
 'FANE-row':'o', 'NetMF': 'd'}

FONTSIZE = 'xx-large'

def plot_precision(filename='arxiv_link_pred'):
    '''
    Plots the precision as a function of k.
    '''
    plt.figure(figsize=(10,6))

    log_k = range(0,14)
    if filename == 'blog_link_pred':
        log_k = range(0,15)
    elif filename == 'flickr_link_pred':
        log_k = range(0,19,2)

    k = [ 2**j for j in log_k ]
    num_k = len(k)

    with open('%s.csv' % filename) as f:
        for line in f:
            ls = line.rstrip().split(',')
            model = ls[0]
            scores = [float(x) for x in ls[1:]]
            label = model
            plt.plot(log_k, scores, \
                marker=markers[model], color=colors[model], label=label)

    plt.xlabel("k",fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)


    ticks = [k[x] for x in range(0,len(k),2)]
    tickcoords = [log_k[x] for x in range(0,len(log_k),2)]
    plt.xticks(tickcoords, ticks, fontsize=FONTSIZE)

    plt.legend(loc='best', fontsize=FONTSIZE)
    plt.grid(linestyle='dashed')
    plt.savefig('%s.eps' % filename, format='eps', dpi=1000, bbox_inches='tight')

def plot_bin_precision(filename='arxiv_binned_new'):
    '''
    Plots the precision as a function across in five bins.
    '''
    plt.figure(figsize=(20,8))

    k = [ 2**j for j in range(0,14) ]
    if filename == 'blog_binned_new':
        k = [ 2**j for j in range(0,15) ] # blog
    elif filename == 'flickr_binned_new':
        k = [ 2**j for j in range(0,19,2) ] # flickr

    num_k = len(k)
    bins = ['Bin 0', 'Bin 1', 'Bin 2', 'Bin 3', 'Bin 4']
    data = {}
    for bin in bins:
        data[bin] = {}

    with open('%s.csv' % filename) as f:
        bin_id = 0
        bin = data['Bin 0']
        for line in f:
            if line == '\r\n':
                bin_id += 1
                bin = data['Bin {}'.format(bin_id)]
            else:
                ls = line.rstrip().split(',')
                model = ls[0]
                scores = [float(x) for x in ls[1:]]
                xids = range(bin_id*num_k, bin_id*num_k+len(scores))

                label = model if bin_id == 0 else None
                plt.plot(xids, scores, color=colors[model], marker=markers[model], label=label)
                bin[model] = scores

    xcoords = [(i+1)*num_k for i in range(len(bins)-1)]
    for i in range(len(xcoords)):
        xcoords[i] -= 0.5
    for xc in xcoords:
        plt.axvline(x=xc, linestyle='--')

    ticks = []
    for bin in bins:
        for i in range(0,len(k),4):
            ticks.append(k[i])

    tickcoords = []
    for i in range(len(bins)):
        tickcoords += range(i*num_k, (i+1)*num_k, 4)
    plt.xticks(tickcoords, ticks, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    if filename == 'arxiv_binned_new':
        bin_names=['removed=1\nn=654','removed=2\nn=287','removed=[3,4]\nn=230',\
        'removed=[5,9]\nn=192','removed>=10\nn=90']
        plt.annotate(bin_names[0],xy=(2,0.9), fontsize=FONTSIZE)
        plt.annotate(bin_names[1],xy=(18,0.9), fontsize=FONTSIZE)
        plt.annotate(bin_names[2],xy=(32,0.9), fontsize=FONTSIZE)
        plt.annotate(bin_names[3],xy=(47,0.9), fontsize=FONTSIZE)
        plt.annotate(bin_names[4],xy=(62,0.9), fontsize=FONTSIZE)
        plt.legend(loc='lower right', fontsize=FONTSIZE)
    elif filename == 'blog_binned_new':
        bin_names=['removed=1\nn=1961','removed=[2,3]\nn=2269','removed=[4,7]\nn=2154',\
        'removed=[8,18]\nn=2082','removed>=19\nn=1569']
        plt.annotate(bin_names[0],xy=(3,0.9), fontsize=FONTSIZE)
        plt.annotate(bin_names[1],xy=(18,0.9), fontsize=FONTSIZE)
        plt.annotate(bin_names[2],xy=(34,0.9), fontsize=FONTSIZE)
        plt.annotate(bin_names[3],xy=(49,0.9), fontsize=FONTSIZE)
        plt.annotate(bin_names[4],xy=(66,0.9), fontsize=FONTSIZE)
        plt.legend(loc='lower right', fontsize=FONTSIZE)
    elif filename == 'flickr_binned_new':
        bin_names=['removed=[1,2]\nn=17547','removed=[3,7]\nn=18426','removed=[8,17]\nn=16149', \
        'removed=[18,48]\nn=16234','removed>49\nn=12139']
        plt.annotate(bin_names[0],xy=(0.7,0.8), fontsize=FONTSIZE)
        plt.annotate(bin_names[1],xy=(11,0.8), fontsize=FONTSIZE)
        plt.annotate(bin_names[2],xy=(21,0.45), fontsize=FONTSIZE)
        plt.annotate(bin_names[3],xy=(31,0.8), fontsize=FONTSIZE)
        plt.annotate(bin_names[4],xy=(42,0.8), fontsize=FONTSIZE)
        plt.legend(loc='upper center', fontsize=FONTSIZE)

    plt.xlabel("k",fontsize=FONTSIZE)
    plt.grid(linestyle='dashed')
    plt.savefig('%s.eps' % filename, format='eps', dpi=1000, bbox_inches='tight')

def plot_param_study(filename='blog_param_study'):
    '''
    Plots the precision as a function of alpha for k=1,k=2,k=4.
    '''
    plt.figure(figsize=(10,6))

    alphas = [(y/10.0) for y in range(1,10)]
    log_k = range(0,15)
    k = [ 2**j for j in log_k ]

    k_ = np.zeros((len(alphas),len(k)))
    with open('%s.csv' % filename, 'r') as f:
        i = 0
        for line in f:
            ls = line.rstrip().split(',')
            scores = [float(x) for x in ls[1:]]
            k_[i] = scores
            i += 1
    k_ = k_.T

    for km in range(0,3):
        label = 'k=%d' % k[km]
        plt.plot(alphas, k_[km], label=label, marker='o')

    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel("alpha",fontsize=FONTSIZE)
    plt.legend(loc='best', fontsize=FONTSIZE)
    plt.grid(linestyle='dashed')
    plt.savefig('%s.eps' % filename, format='eps', dpi=1000, bbox_inches='tight')

if __name__ == "__main__":
    # plot_param_study()
    # plot_precision('arxiv_link_pred')
    plot_bin_precision('flickr_binned_new')
