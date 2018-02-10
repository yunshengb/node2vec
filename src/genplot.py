import collections
import numpy as np
import matplotlib.pyplot as plt
from math import log

colors = {'DeepWalk': 'gold', 'SDNE': 'lightskyblue', 'NANE-weighted': 'purple', \
 'LINE(1st)': 'chocolate', 'LINE(2nd)': 'burlywood', 'LINE(1st+2nd)': 'wheat', \
 'Node2Vec': 'yellowgreen', 'NANE-sym':'magenta', 'NANE-mean':'hotpink', 'NetMF': 'c'}  # change this when re-run

markers = {'DeepWalk': 'x', 'SDNE': 's', 'NANE-weighted': '.', \
 'LINE(1st)': 'v', 'LINE(2nd)': '^', 'LINE(1st+2nd)': '<', 'Node2Vec': '*', 'NANE-sym':'H', \
 'NANE-mean':'o', 'NetMF': 'd'}

def plot_precision(filename='arxiv_link_pred'):
    '''
    Plots the precision as a function of k.
    '''
    FONTSIZE = 'xx-large'
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
    plt.yticks([a/10.0 for a in range(0,11,2)], fontsize=FONTSIZE)


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
    FONTSIZE = 25
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
    plt.xticks(tickcoords, ticks, fontsize=22)
    plt.yticks([a/10.0 for a in range(0,11,2)], fontsize=FONTSIZE)

    if filename == 'arxiv_binned_new':
        # bin_names=['removed=1\nn=654','removed=2\nn=287','removed=[3,4]\nn=230',\
        # 'removed=[5,9]\nn=192','removed>=10\nn=90']
        bin_names = ['654 nodes\neach with\n1 edge\nremoved', '287 nodes\neach with\n2 edges\nremoved', \
        '230 nodes\neach with\n3-4 edges\nremoved', '192 nodes\neach with\n5-9 edges\nremoved', \
        '90 nodes\neach with\n>=10 edges\nremoved'
        ]
        plt.annotate(bin_names[0],xy=(3,0.75), fontsize=FONTSIZE)
        plt.annotate(bin_names[1],xy=(18,0.75), fontsize=FONTSIZE)
        plt.annotate(bin_names[2],xy=(32,0.75), fontsize=FONTSIZE)
        plt.annotate(bin_names[3],xy=(47,0.75), fontsize=FONTSIZE)
        plt.annotate(bin_names[4],xy=(62,0.75), fontsize=FONTSIZE)

    elif filename == 'blog_binned_new':
        # bin_names=['removed=1\nn=1961','removed=[2,3]\nn=2269','removed=[4,7]\nn=2154',\
        # 'removed=[8,18]\nn=2082','removed>=19\nn=1569']
        bin_names = ['1961 nodes\neach with\n1 edge\nremoved', '2269 nodes\neach with\n2-3 edges\nremoved', \
        '2154 nodes\neach with\n4-7 edges\nremoved', '2082 nodes\neach with\n8-18 edges\nremoved', \
        '1569 nodes\neach with\n>=19 edges\nremoved'
        ]
        plt.annotate(bin_names[0],xy=(3,0.75), fontsize=FONTSIZE)
        plt.annotate(bin_names[1],xy=(18,0.75), fontsize=FONTSIZE)
        plt.annotate(bin_names[2],xy=(34,0.75), fontsize=FONTSIZE)
        plt.annotate(bin_names[3],xy=(49,0.75), fontsize=FONTSIZE)
        plt.annotate(bin_names[4],xy=(66,0.75), fontsize=FONTSIZE)

    elif filename == 'flickr_binned_new':
        # bin_names=['removed=[1,2]\nn=17547','removed=[3,7]\nn=18426','removed=[8,17]\nn=16149', \
        # 'removed=[18,48]\nn=16234','removed>49\nn=12139']
        bin_names = ['17547 nodes\neach with\n1-2 edges\nremoved', '18426 nodes\neach with\n3-7 edges\nremoved', \
        '16149 nodes\neach with\n8-17 edges\nremoved', '16234 nodes\neach with\n18-48 edges\nremoved', \
        '12139 nodes\neach with\n>=49 edges\nremoved'
        ]
        plt.annotate(bin_names[0],xy=(0.7,0.7), fontsize=FONTSIZE)
        plt.annotate(bin_names[1],xy=(11,0.7), fontsize=FONTSIZE)
        plt.annotate(bin_names[2],xy=(21,0.7), fontsize=FONTSIZE)
        plt.annotate(bin_names[3],xy=(31,0.7), fontsize=FONTSIZE)
        plt.annotate(bin_names[4],xy=(42,0.7), fontsize=FONTSIZE)

    plt.legend(loc='upper right', fontsize=FONTSIZE, bbox_to_anchor=(1.275,1.025))
    plt.xlabel("k",fontsize=FONTSIZE)
    plt.grid(linestyle='dashed')
    plt.savefig('%s.eps' % filename, format='eps', dpi=1000, bbox_inches='tight')

def plot_param_study(filename='blog_param_study'):
    '''
    Plots the precision as a function of alpha for k=1,k=2,k=4.
    '''
    FONTSIZE = 'xx-large'
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

    for km in range(3):
        label = 'k=%d' % k[km]
        plt.plot(alphas, k_[km], label=label, marker='o')

    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel(r'$\alpha$',fontsize=FONTSIZE)
    plt.legend(loc='best', fontsize=FONTSIZE)
    plt.grid(linestyle='dashed')
    plt.savefig('%s.eps' % filename, format='eps', dpi=1000, bbox_inches='tight')

def plot_dim_study(filename='blog_dim_study'):
    '''
    Plots the precision as a function of alpha for k=1,k=2,k=4.
    '''
    FONTSIZE = 'xx-large'
    plt.figure(figsize=(10,6))

    dimensions = range(100,275,25)
    log_k = range(0,15)
    k = [ 2**j for j in log_k ]

    k_ = np.zeros((len(dimensions),len(k)))
    with open('%s.csv' % filename, 'r') as f:
        i = 0
        for line in f:
            ls = line.rstrip().split(',')
            scores = [float(x) for x in ls[1:]]
            k_[i] = scores
            i += 1
    k_ = k_.T

    for km in range(3):
        label = 'k=%d' % k[km]
        plt.plot(dimensions, k_[km], label=label, marker='o')

    plt.xticks(dimensions, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel("Dimensions",fontsize=FONTSIZE)
    plt.legend(loc='best', fontsize=FONTSIZE, bbox_to_anchor=(1.275,1.025))
    plt.grid(linestyle='dashed')
    plt.savefig('%s.eps' % filename, format='eps', dpi=1000, bbox_inches='tight')

if __name__ == "__main__":
    # plot_param_study()
    plot_dim_study()
    # plot_precision('flickr_link_pred')
    # plot_bin_precision('flickr_binned_new')
