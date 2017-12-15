import numpy as np
import argparse
import os
from utils import similarity_scores, precision, precision_alt


c = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Generate link predicition results.")

    parser.add_argument('--embedding', nargs='?', default='arxiv_cleaned',
                        help='Embedding file. File type must be .npy')

    parser.add_argument('--graph', nargs='?', default='arxiv_cleaned',
                        help='Adjacency matrix file. File type must be .npy')

    parser.add_argument('--hidden-links', nargs='?', default='arxiv_cleaned',
                        help='Hidden links file. File type must be .npy')

    return parser.parse_args()


def generate_results(args, ignore_last=False):
    '''
    Collects precision information on a given embedding
    and original graph information.
    '''

    graph = np.load('%s/../graph/%s_hidden.npy' % (c, args.graph))
    hidden_links = np.load('%s/../graph/%s_testlinks.npy' % (c, args.hidden_links))

    for row in hidden_links:
        x = row[0]
        y = row[1]
        graph[x][y] = graph[y][x] = 1
    np.fill_diagonal(graph, 0)

    k_ = [ 2**j for j in range(0,14) ]

    emb = np.load('%s/../emb/%s.npy' % (c, args.embedding))

    if ignore_last:
        # ignore the last node in embedding and in graph (for arxiv)
        m = len(emb)-1
        emb = emb[:m]
        n = len(graph)
        graph = np.delete(graph, n-1, 0)
        graph = np.delete(graph, n-1, 1)

    sim_scores = similarity_scores(emb, method='euclidean')

    precisions = [precision(k, sim_scores, graph) for k in k_]
    str_precisions = [("%f" % x) for x in precisions]
    print '\t'.join(str_precisions)

    return


if __name__ == "__main__":
    args = parse_args()
    generate_results(args)
