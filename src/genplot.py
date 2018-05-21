import collections
import numpy as np
import matplotlib.pyplot as plt
from math import log

colors = {'DeepWalk': 'peru', 'SDNE': 'green', 'NANE': 'red', \
 'LINE(1st)': 'burlywood', 'LINE(2nd)': 'chocolate', 'LINE(1st+2nd)': 'brown', \
 'Node2Vec': 'orange', 'GCN':'cyan', 'GraphSAGE':'lightskyblue', 'NetMF': 'darkorange'}  # change this when re-run

markers = {'DeepWalk': 'x', 'SDNE': 's', 'NANE': 'd', \
 'LINE(1st)': 'v', 'LINE(2nd)': '^', 'LINE(1st+2nd)': '<', 'Node2Vec': '*', 'GCN':'H', \
 'GraphSAGE':'D', 'NetMF': 'o'}

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
                marker=markers[model], color=colors[model], label=label, markersize=10)

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
    FONTSIZE = 23
    if 'blog' in filename:
        plt.figure(figsize=(19.5, 8)) # changed figsize from (21, 8) to (19.5, 8).
    else:
        plt.figure(figsize=(21, 8))

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
            if line == '\r\n' or line == '\n':
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
        bin_names = ['#nodes: 654\navg deg: 6.2\navg whd: 1', '#nodes: 287\navg deg: 10\navg whd: 2', \
        '#nodes: 230\navg deg: 15\navg whd: 3.4', '#nodes: 192\navg deg: 25\navg whd: 6.4', \
        '#nodes: 90\navg deg: 45\navg whd: 13'
        ]
        plt.annotate(bin_names[0],xy=(2,0.85), fontsize=FONTSIZE)
        plt.annotate(bin_names[1],xy=(17,0.85), fontsize=FONTSIZE)
        plt.annotate(bin_names[2],xy=(31,0.85), fontsize=FONTSIZE)
        plt.annotate(bin_names[3],xy=(45,0.85), fontsize=FONTSIZE)
        plt.annotate(bin_names[4],xy=(62,0.85), fontsize=FONTSIZE)

    elif filename == 'blog_binned_new':
        # bin_names=['removed=1\nn=1961','removed=[2,3]\nn=2269','removed=[4,7]\nn=2154',\
        # 'removed=[8,18]\nn=2082','removed>=19\nn=1569']
        bin_names = ['#nodes: 1961\navg deg: 5\navg whd: 1', '#nodes: 2269\navg deg: 12\navg whd: 2.4', \
        '#nodes: 2154\navg deg: 25\navg whd: 5.2', '#nodes: 2082\navg deg: 58\navg whd: 12', \
        '#nodes: 1569\navg deg: 291\navg whd: 58.8'
        ]
        plt.annotate(bin_names[0],xy=(1.5,0.8), fontsize=FONTSIZE)
        plt.annotate(bin_names[1],xy=(17,0.8), fontsize=FONTSIZE)
        plt.annotate(bin_names[2],xy=(32.5,0.8), fontsize=FONTSIZE)
        plt.annotate(bin_names[3],xy=(47.5,0.8), fontsize=FONTSIZE)
        plt.annotate(bin_names[4],xy=(64,0.8), fontsize=FONTSIZE)

    elif filename == 'flickr_binned_new':
        # bin_names=['removed=[1,2]\nn=17547','removed=[3,7]\nn=18426','removed=[8,17]\nn=16149', \
        # 'removed=[18,48]\nn=16234','removed>49\nn=12139']
        bin_names = ['#nodes: 17547\navg deg: 6.3\navg whd: 1.4', '#nodes: 18426\navg deg: 22.6\navg whd: 4.7', \
        '#nodes: 16149\navg deg: 58.7\navg whd: 11.8', '#nodes: 16234\navg deg: 148\navg whd: 29.3', \
        '#nodes: 12139\navg deg: 652.6\navg whd: 126.7'
        ]
        plt.annotate(bin_names[0],xy=(0.5,0.8), fontsize=FONTSIZE)
        plt.annotate(bin_names[1],xy=(10,0.8), fontsize=FONTSIZE)
        plt.annotate(bin_names[2],xy=(20,0.8), fontsize=FONTSIZE)
        plt.annotate(bin_names[3],xy=(30,0.8), fontsize=FONTSIZE)
        plt.annotate(bin_names[4],xy=(40,0.8), fontsize=FONTSIZE)

    plt.legend(loc='lower center', fontsize=FONTSIZE, bbox_to_anchor=(0.5, -0.27), ncol=5)
    plt.xlabel("k",fontsize=FONTSIZE)
    plt.grid(linestyle='dashed')
    plt.savefig('%s.eps' % filename, format='eps', dpi=1000, bbox_inches='tight')

def plot_param_study(filename='blog_param_study'):
    '''
    Plots the precision as a function of alpha for k=1,k=2,k=4.
    '''
    FONTSIZE = 23
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

    primary_colors = ['red','blue','green']
    study_markers = ['D','x','s']
    for km in range(3):
        label = 'k=%d' % k[km]
        color = primary_colors[km]
        marker = study_markers[km]
        plt.plot(alphas, k_[km], label=label, color=color, marker=marker, markersize=15)

    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=25)
    plt.xlabel(r'$\alpha$',fontsize=35)
    plt.legend(loc='lower right', fontsize=20)
    plt.grid(linestyle='dashed')
    plt.savefig('%s.eps' % filename, format='eps', dpi=1000, bbox_inches='tight')

def plot_dim_study(filename='blog_dim_study'):
    '''
    Plots the precision as a function of alpha for k=1,k=2,k=4.
    '''
    FONTSIZE = 23
    plt.figure(figsize=(10,6))

    dimensions = range(50,275,25)
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

    primary_colors = ['red','blue','green']
    study_markers = ['D','x','s']
    for km in range(3):
        label = 'k=%d' % k[km]
        color = primary_colors[km]
        marker = study_markers[km]
        plt.plot(dimensions, k_[km], label=label, color=color, marker=marker, markersize=15)

    plt.xticks(dimensions, fontsize=FONTSIZE)
    plt.yticks(fontsize=25)
    plt.xlabel("#dimensions",fontsize=35)
    plt.legend(loc='lower right', fontsize=20)
    plt.grid(linestyle='dashed')
    plt.savefig('%s.eps' % filename, format='eps', dpi=1000, bbox_inches='tight')

if __name__ == "__main__":
    # plot_param_study()
    # plot_dim_study()
    plot_precision('arxiv_link_pred')
    plot_precision('blog_link_pred')
    plot_precision('flickr_link_pred')
    plot_bin_precision('arxiv_binned_new')
    plot_bin_precision('blog_binned_new')
    plot_bin_precision('flickr_binned_new')
