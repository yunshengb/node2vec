from utils import convert_embed_to_np
from main import parse_args
import argparse
from math import ceil, floor
import networkx as nx
import numpy as np
import node2vec
import os
from random import shuffle
from gensim.models import Word2Vec

c = os.path.dirname(os.path.realpath(__file__))

def read_graph(remove_self_loops=False):
    '''
    Reads input and hides percent_hidden edges from each node.
    Builds Graph from remaining edges in edgelist.
    '''
    filename = '%s/../graph/%s.%s' % (c, args.input_name, args.input_type)

    if args.input_type == "edgelist":
        if not args.weighted:
            G = nx.read_edgelist(filename, nodetype=int, create_using=nx.DiGraph())
            print '@@@', G
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1

    G = G.to_undirected()

    if nx.is_connected(G):
        return G

    # if the graph is not connected
    # connect the graph using a magic node
    connected_components = nx.connected_components(G)
    magic_node = nx.number_of_nodes(G)+1
    G.add_node(magic_node)

    for comp in connected_components:
        # get highest degree node in component
        root = sorted(comp, key=nx.degree(G), reverse=True)[0]
        # add edge from magic node to root node
        G.add_edge(magic_node, root)
        G[magic_node][root]['weight'] = 1

    # remove self loops
    if remove_self_loops:
        for edge in nx.selfloop_edges(G):
            G.remove_edge(*edge)

    return G

def graph_with_edges_hidden(G, percent_hidden):
    span_tree_edges = set()
    for edge in nx.minimum_spanning_edges(G, data=False):
        span_tree_edges.add(edge)
        span_tree_edges.add((edge[1],edge[0]))

    n_edges = G.number_of_edges()
    hidden_edges = []

    nodes = list(nx.nodes(G))

    for u in nodes:
        neighbors = list(nx.all_neighbors(G, u))
        n_neighbors = len(neighbors)
        edges_to_hide = int(ceil(n_neighbors * percent_hidden))
        shuffle(neighbors)
        for v in neighbors:
            if edges_to_hide > 0 and not (u,v) in span_tree_edges and u!=v:
                G.remove_edge(u, v)
                hidden_edges.append([u,v])
                edges_to_hide -= 1

    print '%d edges hidden out of %d edges -- %f' % \
    (len(hidden_edges), n_edges, len(hidden_edges)/float(n_edges))

    adj_mat = nx.to_numpy_matrix(G)
    outfile = '%s/../graph/%s_hidden.npy' % (c, args.input_name)
    np.save(outfile, adj_mat)

    hidden = np.array(hidden_edges)
    testfile = '%s/../graph/%s_testlinks.npy' % (c, args.input_name)
    np.save(testfile, hidden)

    return G, hidden_edges


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    print 'Number of walks', len(walks)
    print 'An example walk', walks[7]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size,
                     min_count=0, sg=1, workers=args.workers, iter=args.iter)
    emb_file = '%s/../emb/%s_iter_%s.emb' % (c, args.input_name, args.iter)
    model.wv.save_word2vec_format(emb_file)
    print 'args.window_size', args.window_size
    convert_embed_to_np(emb_file, '%s/../emb/%s_emb_iter_%s_p_%s_q_%s_walk_%s_win_%s.npy' % \
        (c, args.input_name, args.iter, args.p, args.q, args.num_walks, args.window_size), ignore_last=True)

    return


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    print 'Reading graph'
    graph = read_graph(remove_self_loops=False)
    nx_G, hidden_edges = graph_with_edges_hidden(graph, 0.01)
    print 'Creating node2vec graph'
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    print 'Preprocessing'
    G.preprocess_transition_probs()
    print 'Generating walks'
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    print 'Learning embeddings'
    learn_embeddings(walks)


if __name__ == "__main__":
    args = parse_args()
    main(args)
