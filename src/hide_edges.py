import argparse
from math import ceil, floor
import networkx as nx
import numpy as np
import os
from random import shuffle

c = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Clean graph and hide edges from graph.")

    parser.add_argument('--input_name', nargs='?', default='arxiv',
                        help='Adjacency matrix file. File type must be .npy')

    return parser.parse_args()

def read_graph():
    '''
    Reads input and builds graph. If the graph is not connected,
    adds a magic node that connects all components
    via the node with the highest degree.
    '''
    filename = '%s/../graph/%s.edgelist' % (c, args.input_name)

    G = nx.read_edgelist(filename, nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    G = G.to_undirected()

    # remove edges to self
    for edge in nx.selfloop_edges(G):
        G.remove_edge(edge[0],edge[1])

    # remove all isolates
    for node in nx.isolates(G):
        G.remove_node(node)

    # relabel nodes
    G = nx.convert_node_labels_to_integers(G)

    if nx.is_connected(G):
        return G

    # if the graph is not connected
    # connect the graph using a magic node
    connected_components = nx.connected_components(G)
    magic_node = nx.number_of_nodes(G)
    G.add_node(magic_node)

    for comp in connected_components:
        # get highest degree node in component
        root = sorted(comp, key=nx.degree(G), reverse=True)[0]
        # add edge from magic node to root node
        G.add_edge(magic_node, root)
        G[magic_node][root]['weight'] = 1

    return G

def graph_with_edges_hidden(G, percent_hidden):
    '''
    Given a graph, hides a percentage of edges per node
    while maintaining a connected graph result.
    '''
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
        edges_to_hide = int(floor(n_neighbors * percent_hidden))
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


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    print 'Reading graph'
    graph = read_graph()
    nx_G, hidden_edges = graph_with_edges_hidden(graph, 0.16)


if __name__ == "__main__":
    args = parse_args()
    main(args)
