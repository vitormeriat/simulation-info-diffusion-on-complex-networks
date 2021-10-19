import EoN
import enum
import random
import os.path
import requests
import networkx as nx
from time import time
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import defaultdict

#from azure.storage.queue import QueueClient, TextBase64EncodePolicy
from tqdm import tqdm
import logging

class Simulation():
    def rate_function(self, G, node, status, parameters):
        tau,gamma = parameters
        if status[node] == 'I':
            return gamma
        elif status[node] == 'S':
            return tau*len([nbr for nbr in G.neighbors(node) if status[nbr] == 'I'])
        else:
            return 0

    def transition_choice(self, G, node, status, parameters):
        if status[node] == 'I':
            return 'R'
        elif status[node] == 'S':
            return 'I'

    def get_influence_set(self, G, node, status, parameters):
        return {nbr for nbr in G.neighbors(node) if status[nbr] == 'S'}
    
    def fast_sir(self, G, beta, gamma, R_0, tmax):
        sim = EoN.fast_SIR(G, tau=beta, gamma=gamma, rho=R_0, return_full_data=True, tmax=tmax)
        summary = sim.summary()
        return sim, summary
    
    def complex_contagion(self, G, beta, gamma, tmax):
        IC = defaultdict(lambda: 'S')
        for node in range(2):
            IC[random.choice(list(G.nodes))] = 'I'

        sim = EoN.Gillespie_complex_contagion(
            G, 
            self.rate_function,
            self.transition_choice, 
            self.get_influence_set, 
            IC,
            return_statuses=('S', 'I', 'R'),
            parameters=(beta, gamma), 
            return_full_data=True,
            tmax=tmax)
        summary = sim.summary()
        return sim, summary
    
class Stats():
    
    def __init__(self):
        self.round_len = 5
    
    def log_time(self, msg, start_time):
        end = time()
        elapsed = end - start_time
        print('{:>40s}     {:}'.format(msg, elapsed))

    def density(self, G):
        start_time = time()
        try:
            density = round(nx.density(G), self.round_len)
        except nx.NetworkXError as e:
            density = str(e)
        self.log_time('Density', start_time)
        return density

    def radius(self, G):
        start_time = time()
        try:
            radius = round(nx.radius(G), self.round_len)
        except nx.NetworkXError as e:
            radius = str(e)
        self.log_time('Radius', start_time)
        return radius

    def diameter(self, G):
        start_time = time()
        try:
            diameter = round(nx.diameter(G), self.round_len)
        except nx.NetworkXError as e:
            diameter = str(e)
        self.log_time('Diameter', start_time)
        return diameter

    def average_shortest_path(self, G):
        start_time = time()
        try:
            average_shortest_path = round(
                nx.average_shortest_path_length(G), self.round_len)
        except nx.NetworkXError as e:
            average_shortest_path = str(e)
        self.log_time('Average shortest path length', start_time)
        return average_shortest_path

    def average_degree(self, G):
        start_time = time()
        G_deg = nx.degree_histogram(G)
        G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
        average_degree = round(
            sum(G_deg_sum) / G.number_of_nodes(), self.round_len)
        self.log_time('Average Degree', start_time)
        return average_degree

    def betweenness_centrality(self, G):
        # Run betweenness centrality
        start_time = time()
        k = int(float(G.number_of_nodes()) / 100 * float(10))
        #betweenness_dict = nx.betweenness_centrality(self.G)
        betweenness_dict = nx.betweenness_centrality(G, k=k)
        self.log_time('Betweenness centrality', start_time)
        return betweenness_dict

    def closeness_centrality(self, G, _is):
        # Run closeness centrality
        start_time = time()
        closeness_dict = {}
        for g in _is:
            closeness_dict[g] = nx.closeness_centrality(G, u=g)
        self.log_time('Closeness centrality', start_time)
        return closeness_dict

    def degree_centrality(self, G):
        # Run degree centrality
        start_time = time()
        degree_dict = nx.degree_centrality(G)
        self.log_time('Degree centrality', start_time)
        return degree_dict

    def eigenvector_centrality(self, G):
        # Run eigenvector centrality
        start_time = time()
        eigenvector_dict = nx.eigenvector_centrality(G, max_iter=1000)
        self.log_time('Eigenvector centrality', start_time)
        return eigenvector_dict

    def clustering_coefficient(self, G):
        # Run clustering coefficient
        start_time = time()
        clustering_dict = nx.clustering(G)
        self.log_time('Clustering coefficient', start_time)
        return clustering_dict
    
    def run_stats(self, betweenness, closeness, degree, eigenvector, clustering, _is):
        stats = []
        for i in _is:
            stats.append({
                'node': i,
                'degree_centrality': round(degree[i], self.round_len),
                'closeness_centrality': round(closeness[i], self.round_len),
                'betweenness_centrality': round(betweenness[i], self.round_len),
                'eigenvector_centrality': round(eigenvector[i], self.round_len),
                'clustering_coefficient': round(clustering[i], self.round_len),
            })
        return stats



# ========================================================================================================


class EnumResult():
    def get(self, msg):
        return msg.value

@enum.unique
class BaseModel(enum.Enum):
    FACEBOOK_SIMMONS='facebook_simmons81'
    FACEBOOK_BOWDOIN='facebook_bowdoin47'
    FACEOOBK_HAVERFORD='facebook_haverford76'
    FACEBOOK_EGO='facebook_ego'

    TWITTER_EGO='twitter_ego'
    TWITTER_SOC='twitter_soc'
    YOUTUBE_SOC='youtube_soc'

    ERDOS_RENYI='erdos_renyi_gnp'
    BARABASI='barabasi_albert'
    WATTS_STROGATZ='watts_strogatz'

@enum.unique
class SimulationType(enum.Enum):
    FAST_SIR='fast_sir'
    COMPLEX_CONTAGION='complex_contagion'


# ========================================================================================================


def get_initial_infecteds(simulation):
    try:
        return [i for i in range(simulation.G.number_of_nodes()) if simulation.node_status(i, 0) == 'I']
    except Exception:
        return None


def get_tmax(sim):
    lst = list(sim[1]['I'])
    element = int(max(lst))
    return element, lst.index(element)


def statistics(G, sim, summary):
    stats = Stats()
    _is = get_initial_infecteds(sim)
    print(f'Nós iniciais infectados: {_is}')

    print("\n%40s    %s" % ('Process', 'Time'))
    print("%40s    %s" % ('='*40, '='*25))

    calculated_statistics = None

    if _is != None:

        # Run betweenness centrality
        betweenness_dict = stats.betweenness_centrality(G)
        # Run closeness centrality
        closeness_dict = stats.closeness_centrality(G, _is)
        # Run degree centrality
        degree_dict = stats.degree_centrality(G)
        # Run eigenvector centrality
        eigenvector_dict = stats.eigenvector_centrality(G)
        # Run clustering coefficient
        clustering_dict = stats.clustering_coefficient(G)
        
        calculated_statistics = stats.run_stats(betweenness_dict, closeness_dict, degree_dict, eigenvector_dict, clustering_dict, _is)

    element, tmax = get_tmax(summary)

    result = {
        'max_infection_time': tmax,
        'max_infection_len': element,
        'T0': {
            'S': int(summary[1]['S'][0]),
            'I': int(summary[1]['I'][0])
        },
        'TMAX': {
            'S': int(summary[1]['S'][-1:][0]),
            'I': int(summary[1]['I'][-1:][0]),
            'R': int(summary[1]['R'][-1:][0])
        },
        'stats': calculated_statistics
    }

    return result

def print_network_stats(net_stats):
    network_stats = [{
        'max_infection_time': net_stats['max_infection_time'],
        'max_infection_len': net_stats['max_infection_len'],
        'T0.S': net_stats['T0']['S'],
        'T0.I': net_stats['T0']['I'],
        'TMAX.S': net_stats['TMAX']['S'],
        'TMAX.I': net_stats['TMAX']['I'],
        'TMAX.R': net_stats['TMAX']['R']
    }]

    print(f'\n{tabulate(network_stats, headers="keys", tablefmt="github")}')
    
def print_node_stats(net_stats):
    node_stats = []
    for n in net_stats['stats']:
        node_stats.append({
            'node': n['node'],
            'degree_centrality': n['degree_centrality'],
            'closeness_centrality': n['closeness_centrality'],
            'betweenness_centrality': n['betweenness_centrality'],
            'eigenvector_centrality': n['eigenvector_centrality'],
            'clustering_coefficient': n['clustering_coefficient']
        })

    print(f'\n{tabulate(node_stats, headers="keys", tablefmt="github")}')
    
def network_statistics(G, show=True):
    stats = Stats()
    
    print('Métricas iniciais da rede...')
    print(f'\n{nx.info(G)}')
    print("\n%40s    %s" % ('Process', 'Time'))
    print("%40s    %s" % ('='*40, '='*25))

    # Run density, diameter & radius network
    density = stats.density(G)
    diameter = stats.diameter(G)
    radius = stats.radius(G)
    # Run calculating average shortest path length
    #average_shortest_path = stats.average_shortest_path(G)
    # Run average degree
    average_degree = stats.average_degree(G)

    try:
        is_connected = nx.is_connected(G)
    except Exception as e:  # nx.NetworkXError as e:
        is_connected = str(e)

    try:
        number_connected_components = nx.number_connected_components(G)
    except Exception as e:  # nx.NetworkXError as e:
        number_connected_components = str(e)

    metric = {
        'number_of_nodes': G.number_of_nodes(),
        'number_of_edges': G.number_of_edges(),
        'density': density,
        'diameter':diameter,
        'radius':radius,
        #'average_shortest_path':average_shortest_path,
        'average_degree':average_degree,
        'is_connected':is_connected,
        'number_connected_components':number_connected_components
    }
    
    #print(f'\n{tabulate([metric], headers="keys", tablefmt="github")}')
    
    return metric