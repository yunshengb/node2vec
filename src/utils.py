import numpy as np
import sys
from scipy.spatial.distance import pdist, squareform

def convert_embed_to_np(emb_file, np_file, ignore_last=False):
    print 'Convert %s to %s' % (emb_file, np_file)
    with open(emb_file) as f:
        lines = f.readlines()
    t = lines[0].split()
    s = m = int(t[0])
    n = int(t[1])

    if ignore_last:
        s = m-1

    mat = np.zeros((s, n))
    for line in lines[1:]:
        t = line.rstrip().split()
        r = int(t[0])

        li = [ float(x) for x in t[1:] ]

        if r != s or not ignore_last:
            mat[r] = li

    np.save(np_file, mat)


def format_edgelist(edg_file):
    '''
    Converts an edgelist with arbitrarily labelled vertices
    to edgelist with vertices labeled 0...<num_vertices-1>
    '''
    with open(edg_file) as f:
        lines = f.readlines()

    # grab all vertices and sort them
    temp = []
    for line in lines:
        str_v = line.split()
        temp.append(int(str_v[0]))
        temp.append(int(str_v[1]))
    vertices = np.unique(temp)

    # create mapping of original vertex
    # labels to 0...<num_vertices-1>
    mapping = {}
    for x in range(len(vertices)):
        mapping[vertices[x]] = x

    # convert edgelist with new labels
    outfile = '%s_cleaned.edgelist' % edg_file.split('.')[0]
    with open(outfile, 'w') as f:
        for line in lines:
           str_v = line.split()
           output = '%d\t%d\n' % (mapping[int(str_v[0])], mapping[int(str_v[1])])
           f.write(output)


def similarity_scores(emb, method):
    '''
    Returns similarity scores for each pair in network
    given network embedding as numpy array.
    '''
    if method == 'euclidean':
        sim_array = 1 / pdist(emb)
        sim_mat = squareform(sim_array)
        np.fill_diagonal(sim_mat, sys.maxint)
        return sim_mat
    elif method=='dot':
        return np.dot(emb, emb.T)
    else:
        print "ERROR: Did not recognize method name. Use 'euclidean' or 'dot'."
        raise SystemExit


def precision_alt(k, sim_scores, graph):
    '''
    Returns the precision at k for all links in original graph.
    '''
    np.fill_diagonal(sim_scores, 0)
    cond_sim = squareform(sim_scores)
    sort_cond_sim = np.argsort(-cond_sim)
    truncation = sort_cond_sim[:k]

    true_labels = np.nonzero(squareform(graph))[0]
    matches = len(np.intersect1d(truncation, true_labels, assume_unique=True))
    precision = matches/float(k)
    print precision


def precision(k, sim_scores, graph, degree_range=None):
    '''
    Returns the precision at k at each node in original graph
    and takes the average.
    If given degree cutoff, calculate precision on nodes
    with degrees given or more.
    '''

    n = len(sim_scores)
    m = 0 # running count of nodes tested
    # sort sim_scores along row axis and obtain indices
    np.fill_diagonal(sim_scores, 0)

    precision = 0.0
    for i in range(n):
        z = np.count_nonzero(graph[i])

        if (degree_range == None and z > 0) or \
        (degree_range != None and z in degree_range):
            true_labels = np.nonzero(graph[i])[0]
            sorted_v = np.argsort(-sim_scores[i])
            truncation = sorted_v[:k]
            intersection = np.intersect1d(truncation, true_labels, assume_unique=True)
            precision_i = float(len(intersection)) / k
            precision += precision_i
            m += 1

    return precision / m

