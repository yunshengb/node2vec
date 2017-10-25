import numpy as np
import argparse
import os
from utils import similarity_scores, precision

c = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Generate link predicition results.")

    parser.add_argument('--embedding', nargs='?', default='arxiv_cleaned',
                        help='Path to embedding. File type must be .npy')

    parser.add_argument('--hidden-links', nargs='?', default='arxiv_cleaned',
                        help='Path to hidden links. File type must be .npy')

    return parser.parse_args()


def generate_results(args):
    emb = np.load('%s/../emb/%s_emb_iter_1_p_1_q_1_walk_40_win_10.npy' % (c, args.embedding))
    hidden_links = np.load('%s/../graph/%s_testlinks.npy' % (c, args.hidden_links))

    l2_sim_scores = similarity_scores(emb, method='euclidean')
    cos_sim_scores = similarity_scores(emb, method='cosine')

    k_ = [2, 10, 100, 200, 300, 500, 800, 1000, 10000]

    for k in k_:
        print 'l2 precision@k=%d: %f' % (k, precision(k, l2_sim_scores, hidden_links))

    for k in k_:
        print 'cos precision@k=%d: %f' % (k, precision(k, cos_sim_scores, hidden_links))


if __name__ == "__main__":
    args = parse_args()
    generate_results(args)
