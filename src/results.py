import numpy as np
import argparse
import os
from utils import similarity_scores, precision, precision_alt, precision_rm
import scipy.io

c = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Generate link predicition results.")

    parser.add_argument('--embedding', nargs='?', default='blog',
                        help='Embedding file. File type must be .npy')

    parser.add_argument('--graph', nargs='?', default='blog',
                        help='Adjacency matrix file. File type must be .npy')

    parser.add_argument('--hidden-links', nargs='?', default='blog',
                        help='Hidden links file. File type must be .npy')

    return parser.parse_args()


def generate_results(args, ignore_last=False):
    '''
    Collects precision information on a given embedding
    and original graph information.
    '''

    graph = np.load('%s/../graph/%s_hidden.npy' % (c, args.graph))
    hidden_links = np.load('%s/../graph/%s_testlinks.npy' % (c, args.hidden_links))
    emb = np.load('%s/../emb/%s.npy' % (c, args.embedding))
    # file_path = '%s/../emb/%s.mat' % (c,arg.embedding)
    # emb = scipy.io.loadmat(file_path)['embedding']

    if ignore_last:
        # ignore the last node in embedding and in graph (for arxiv)
        m = len(emb)-1
        emb = emb[:m]
        n = len(graph)-1
        graph = np.delete(graph, n, 0)
        graph = np.delete(graph, n, 1)

    test_mat = np.zeros((len(graph),len(graph)))
    for row in hidden_links:
        x = row[0]
        y = row[1]
        graph[x][y] = graph[y][x] = 1
        test_mat[x][y] = test_mat[y][x] = 1
    np.fill_diagonal(graph, 0)

    sim_scores = similarity_scores(emb, method='dot')

    k_ = [ 2**j for j in range(0,14) ] # use for arxiv
    # k_ = [ 2**j for j in range(0,15) ] # use for blog
    # k_ = [ 2**j for j in range(0,19,2) ] # use for flickr

    # uncomment these lines for binned results
    # ranges = [range(1,2), range(2,3), range(3,4), range(4,5), range(5,15)] # use for arxiv
    # ranges = [range(1,2), range(2,4), range(4,8), range(8,19), range(19,839)] # use for blog
    # ranges = [range(1,3), range(3,8), range(8,18), range(18,49), range(49,1178)] # use for flickr
    # for n_links in ranges:
    # precisions = [precision_rm(k, sim_scores, graph, test_mat, n_links) for k in k_]

    precisions = [precision(k, sim_scores, graph) for k in k_]
    str_precisions = [("%f" % x) for x in precisions]
    print '\t'.join(str_precisions)

    return


if __name__ == "__main__":
    args = parse_args()
    generate_results(args) # set ignore_last=True for arxiv