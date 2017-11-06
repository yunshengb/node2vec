import numpy as np
import argparse
import os
from utils import similarity_scores, precision

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

    emb = np.load('%s/../emb/%s.npy' % (c, args.embedding))
    if ignore_last:
        m = len(emb)-1
        emb = emb[:m]

    sim_scores = similarity_scores(emb, method='dot')
    graph = np.load('%s/../graph/%s_hidden.npy' % (c, args.graph))
    hidden_links = np.load('%s/../graph/%s_testlinks.npy' % (c, args.hidden_links))

    for row in hidden_links:
        x = row[0]
        y = row[1]
        graph[x][y] = graph[y][x] = 1
    np.fill_diagonal(graph, 0)

    k_ = [2, 10, 100, 200, 300, 500, 800, 1000, 10000]
    precisions = [precision(k, sim_scores, graph) for k in k_]
    str_precisions = [("%f" % x) for x in precisions]
    print '\t'.join(str_precisions)

    return


if __name__ == "__main__":
    args = parse_args()
    generate_results(args)
