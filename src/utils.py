import numpy as np
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
        np.fill_diagonal(sim_mat, 1)
        return sim_mat
    elif method == 'cosine':
        dist_array = pdist(emb, 'cosine')
        dist_mat = squareform(dist_array)
        sim_mat = 1 - dist_mat
        return sim_mat
    elif method=='dot':
        return np.dot(emb, emb.T)
    else:
        print "ERROR: Did not recognize method name. Use 'euclidean' or 'cosine'."
        raise SystemExit


def precision_predicted(k, sim_scores, hidden_links):
    '''
    Returns the precision at k for predicted links
    given a matrix of similarity scores and array of hidden links.
    '''

    n = len(sim_scores)
    mask = np.zeros((n, n))
    for row in hidden_links:
        x = row[0]
        y = row[1]
        mask[x][y] = mask[y][x] = 1
    np.fill_diagonal(mask, 0)

    # sort sim_scores along row axis and obtain indices
    np.fill_diagonal(sim_scores, 0)
    sorted_v = np.argsort(-sim_scores, axis=1)

    precision = 0.0
    for i in range(n):
        truncation = sorted_v[i][:k]
        true_labels = np.nonzero(mask[i])[0]
        intersection = np.intersect1d(truncation, true_labels, assume_unique=True)
        # print intersection
        precision_i = float(len(intersection)) / k
        precision += precision_i

    m = np.count_nonzero(np.any(mask, axis=1))
    return precision / m


def precision(k, sim_scores, graph):
    '''
    Returns the precision at k given original graph.
    '''

    n = len(graph)
    # sort sim_scores along row axis and obtain indices
    np.fill_diagonal(sim_scores, 0)
    sorted_v = np.argsort(-sim_scores, axis=1)

    precision = 0.0
    for i in range(n):
        truncation = sorted_v[i][:k]
        true_labels = np.nonzero(graph[i])[0]
        intersection = np.intersect1d(truncation, true_labels, assume_unique=True)
        precision_i = float(len(intersection)) / k
        precision += precision_i

    m = np.count_nonzero(np.any(graph, axis=1))
    return precision / m

