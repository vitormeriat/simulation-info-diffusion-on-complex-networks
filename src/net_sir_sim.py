import os
import EoN
import sys
import uuid
import enum
import random
import platform
from time import time
import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt
from pymongo import MongoClient
from collections import defaultdict

base_path = 'simulation_result' if platform.system(
) == 'Linux' else 'C:/simulation_result'


@enum.unique
class ProcessStatus(enum.IntEnum):
    success = 0     # Indicates successful program completion.
    failure = 101   # Indicates unsuccessful program completion in a general sense


class DataBase():

    uri = "mongodb://root:apzVz53vKVcF@191.238.212.204"

    def __init__(self):
        pass

    def connect_db(self):
        self.conn = MongoClient(self.uri, connect=False)
        self.db = self.conn['graph']

    def this_simulation_exists(self, sim_name):
        self.connect_db()
        coll = self.db['summary']
        item = coll.find_one(
            {"simulation_name": sim_name}, {'_id'})
        self.conn.close()
        return item is not None

    def insert_summary(self, summary):
        self._extracted_from_insert_simulation_2('summary', summary)

    def insert_simulation(self, simulation):
        self._extracted_from_insert_simulation_2('simulation', simulation)

    # TODO Rename this here and in `insert_summary` and `insert_simulation`
    def _extracted_from_insert_simulation_2(self, arg0, arg1):
        self.connect_db()
        coll = self.db[arg0]
        coll.insert_one(arg1)
        self.conn.close()


class Base():

    def __init__(self):
        self.start_time = 0
        self.round_len = 5

    def end_log(self):
        end = time()
        return end - self.start_time

    def log_with_time(self, msg, is_start=True):

        if is_start:
            self.start_time = time()
            print(msg)
        else:
            elapsed = self.end_log()
            print('{:>40s}     {:}'.format(msg, elapsed))

    def query_yes_no(self, question, default="yes"):
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write(
                    "Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


class Simulation():

    def __init__(self, num_simulations, simulation_name):
        self.num_simulations = num_simulations
        self.simulation_name = simulation_name

    def generate_folder(self, path):
        os.mkdir(path)

    def initialize(self, graph):
        db = DataBase()
        if db.this_simulation_exists(self.simulation_name):
            print(f'\n{"═"*100}\n')
            print(f'Simulation \
                {self.simulation_name} already exists. Enter another name.')
            print(f'\n{"═"*100}\n')
            return False
        else:
            self.graph = graph
            self.generate_folder(
                f'{base_path}/{self.simulation_name}')
            return True

    def simulate(self):
        start_time_global = time()
        self.graph.simulation_name = self.simulation_name

        for i in range(self.num_simulations):
            start_time = time()

            print(f'\n{"═"*100}\nIniciando simulação \
                {i+1} de {self.num_simulations}: {self.simulation_name}\n')

            path = f"{base_path}/{self.simulation_name}/{f'{i}'.rjust(10,'0')}"
            self.generate_folder(path)
            self.graph.path = path
            self.graph.iteration_number = i+1
            # self.graph.edges_configuration()
            self.graph.simulation_in_graph()

            end_time = time()
            print(f'\nSimulação {i+1} concluída em \
                {end_time - start_time} seconds')

        end_time_global = time()

        elapsed = end_time_global - start_time_global

        print(f'\n{"█"*100}\n\n')
        print('Finalizando a simulação e gravando a sumarização...')

        #print("\n%40s    %s" % ('General Process', 'Time'))
        #print("%40s    %s" % ('='*40, '='*25))

        self.graph.save_summary(elapsed, self.num_simulations)

        print(f'\nSimulação {self.simulation_name} concluída em\
            {elapsed} seconds\n')


class Stats(Base):

    def __init__(self):
        super().__init__()

    def density(self, G):
        self.start_time = time()
        try:
            density = round(nx.density(G), self.round_len)
        except nx.NetworkXError as e:
            density = str(e)
        self.log_with_time('Density', False)
        return density

    def radius(self, G):
        self.start_time = time()
        try:
            radius = round(nx.radius(G), self.round_len)
        except nx.NetworkXError as e:
            radius = str(e)
        self.log_with_time('Radius', False)
        return radius

    def diameter(self, G):
        self.start_time = time()
        try:
            diameter = round(nx.diameter(G), self.round_len)
        except nx.NetworkXError as e:
            diameter = str(e)
        self.log_with_time('Diameter', False)
        return diameter

    def average_shortest_path(self, G):
        self.start_time = time()
        try:
            average_shortest_path = round(
                nx.average_shortest_path_length(G), self.round_len)
        except nx.NetworkXError as e:
            average_shortest_path = str(e)
        self.log_with_time('Average shortest path length', False)
        return average_shortest_path

    def average_degree(self, G):
        self.start_time = time()
        G_deg = nx.degree_histogram(G)
        G_deg_sum = [a * b for a, b in zip(G_deg, range(len(G_deg)))]
        average_degree = round(
            sum(G_deg_sum) / G.number_of_nodes(), self.round_len)
        self.log_with_time('Average Degree', False)
        return average_degree

    def betweenness_centrality(self, G):
        # Run betweenness centrality
        self.start_time = time()
        k = int(float(G.number_of_nodes()) / 100 * float(10))
        #betweenness_dict = nx.betweenness_centrality(self.G)
        betweenness_dict = nx.betweenness_centrality(G, k=k)
        self.log_with_time('Betweenness centrality', False)
        return betweenness_dict

    def closeness_centrality(self, G, _is):
        # Run closeness centrality
        self.start_time = time()
        closeness_dict = {g: nx.closeness_centrality(G, u=g) for g in _is}
        self.log_with_time('Closeness centrality', False)
        return closeness_dict

    def degree_centrality(self, G):
        # Run degree centrality
        self.start_time = time()
        degree_dict = nx.degree_centrality(G)
        self.log_with_time('Degree centrality', False)
        return degree_dict

    def eigenvector_centrality(self, G):
        # Run eigenvector centrality
        self.start_time = time()
        eigenvector_dict = nx.eigenvector_centrality(G, max_iter=1000)
        self.log_with_time('Eigenvector centrality', False)
        return eigenvector_dict

    def clustering_coefficient(self, G):
        # Run clustering coefficient
        self.start_time = time()
        clustering_dict = nx.clustering(G)
        self.log_with_time('Clustering coefficient', False)
        return clustering_dict


class Graph(Base):

    def __init__(self):
        super().__init__()

        self.netowrk_model = None
        self.path = None
        self.simulation_name = None
        self.calculated_statistics = None
        self.is_real_network = False
        self.iteration_number = 0

    def print_configuration_result(self):

        self.stats = Stats()
        print('Métricas iniciais da rede...')

        print(f'\n{nx.info(self.G)}')

        print("\n%40s    %s" % ('Process', 'Time'))
        print("%40s    %s" % ('='*40, '='*25))

        # Run density, diameter & radius network
        self.density = self.stats.density(self.G)
        self.diameter = self.stats.diameter(self.G)
        self.radius = self.stats.radius(self.G)
        # Run calculating average shortest path length
        self.average_shortest_path = self.stats.average_shortest_path(self.G)
        # Run average degree
        self.average_degree = self.stats.average_degree(self.G)

        try:
            is_connected = nx.is_connected(self.G)
        except Exception as e:  # nx.NetworkXError as e:
            is_connected = str(e)

        try:
            number_connected_components = nx.number_connected_components(
                self.G)
        except Exception as e:  # nx.NetworkXError as e:
            number_connected_components = str(e)

        table = [
            {'metric': 'Density', 'result': self.density},
            {'metric': 'Diameter', 'result': self.diameter},
            {'metric': 'Radius', 'result': self.radius},
            {'metric': 'Average Shortest_path',
                'result': self.average_shortest_path},
            {'metric': 'Average Degree', 'result': self.average_degree},
            {'metric': 'This Graph is connected?', 'result': is_connected},
            {'metric': 'Number of different connected components',
                'result': number_connected_components}
        ]
        print(f'\n{tabulate(table, headers="keys", tablefmt="github")}')

    def configuration(self, network, gamma, beta, R_0, tmax, simulation_type):
        self.G = network
        self.gamma = gamma
        self.beta = beta
        self.R_0 = R_0
        self.tmax = tmax
        self.simulation_type = simulation_type

        self.print_configuration_result()

    def create_net_erdos_renyi(self, nodes, probability):
        self.netowrk_model = 'erdos_renyi_gnp'
        G = nx.gnp_random_graph(nodes, probability)
        isolates = list(nx.isolates(G))
        if isolates:
            return G.remove_nodes_from(list(nx.isolates(G)))
        else:
            return G

    def create_net_barabasi_albert(self, nodes, edges):
        self.netowrk_model = 'barabasi_albert'
        G = nx.barabasi_albert_graph(nodes, edges)
        isolates = list(nx.isolates(G))
        if isolates:
            return G.remove_nodes_from(list(nx.isolates(G)))
        else:
            return G

    def create_net_watts_strogatz(self, nodes, edges, probability):
        self.netowrk_model = 'watts_strogatz'
        G = nx.watts_strogatz_graph(nodes, edges, probability)
        isolates = list(nx.isolates(G))
        if isolates:
            return G.remove_nodes_from(list(nx.isolates(G)))
        else:
            return G

    def create_net_facebook(self, model_id):
        self.is_real_network = True
        #self.netowrk_model = f'facebook-{model_id}'
        #["Ego", "Bowdoin47", "Haverford76", "Simmons81"]
        if model_id == 'Ego':
            self.netowrk_model = 'snap-Ego'
            # return nx.read_edgelist("../data/facebook_combined.txt.gz", create_using=nx.Graph(), nodetype=int)
            return nx.read_gexf("../data/facebook_combined.gexf")
        elif model_id == 'Bowdoin47':
            self.netowrk_model = 'socfb-Bowdoin47'
            # return nx.read_adjlist("../data/socfb-Bowdoin47.mtx", create_using=nx.DiGraph(), nodetype=int)
            return nx.read_gexf("../data/socfb-Bowdoin47.gexf")
        elif model_id == 'Haverford76':
            self.netowrk_model = 'socfb-Haverford76'
            # return nx.read_adjlist("../data/socfb-Haverford76.mtx", create_using=nx.DiGraph(), nodetype=int)
            return nx.read_gexf("../data/socfb-Haverford76.gexf")
        elif model_id == 'Simmons81':
            self.netowrk_model = 'socfb-Simmons81'
            # return nx.read_adjlist("../data/socfb-Simmons81.mtx", create_using=nx.DiGraph(), nodetype=int)
            return nx.read_gexf("../data/socfb-Simmons81.gexf")
        elif model_id == 'Facebook01':
            self.netowrk_model = 'fb-01'
            return nx.read_gexf('../data/fb1.gexf')
        elif model_id == 'Facebook02':
            self.netowrk_model = 'fb-02'
            return nx.read_gexf('../data/fb2.gexf')
        elif model_id == 'Facebook03':
            self.netowrk_model = 'fb-03'
            return nx.read_gexf('../data/fb3.gexf')
        elif model_id == 'Facebook04':
            self.netowrk_model = 'fb-04'
            return nx.read_gexf('../data/fb4.gexf')
        else:
            None

    def create_net_twitter(self):
        self.is_real_network = True
        self.netowrk_model = 'twitter'

        return nx.read_edgelist("../data/twitter_combined.txt.gz", create_using=nx.Graph(), nodetype=int)

    #     return nx.read_edgelist("../data/twitter_combined.txt.gz", create_using=nx.Graph(), nodetype=int)

    def edges_configuration(self):
        self.log_with_time('Iniciando configuração dos nós')

        w = [random.random() for i in range(self.G.number_of_edges())]
        s = max(w)
        w = [i/s for i in w]
        for k, (i, j) in enumerate(self.G.edges()):
            self.G[i][j]['weight'] = w[k]
        labels = {i: i for i in list(self.G.nodes)}
        #labels = [dict({i, i}) for i in list(self.G.nodes)]

        edgewidth = [d['weight'] for (u, v, d) in self.G.edges(data=True)]

        nx_kwargs = {
            "with_labels": True,
            "pos": nx.spring_layout(self.G),
            # "pos": nx.circular_layout(self.G),
            "width": edgewidth,
            "alpha": 0.7,
            "labels": labels
        }
        self.nx_kwargs = nx_kwargs

        self.log_with_time('Configuração dos nós concluída', False)

    def save_sir(self):
        print('Gerando imagem SIR')
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.summary[0], self.summary[1]['S'],
            label="S", linewidth=3, color='g', alpha=0.5)
        plt.plot(
            self.summary[0], self.summary[1]['I'],
            label="I", linewidth=3, color='r', alpha=0.5)
        plt.plot(
            self.summary[0], self.summary[1]['R'],
            label="R", linewidth=3, color='b', alpha=0.5)
        plt.legend()
        plt.savefig(f'{self.path}/sir.png')
        plt.close()

    def get_initial_infecteds(self, simulation):
        try:
            return [i for i in range(simulation.G.number_of_nodes()) if simulation.node_status(i, 0) == 'I']
        except Exception:
            return None

    def save_summary(self, elapsed, num_simulations):

        db = DataBase()
        db.insert_summary({
            '_id': str(uuid.uuid4()),
            'simulation_name': str(self.simulation_name),
            'num_simulations': num_simulations,
            'elapsed_time': round(elapsed, self.round_len),
            'netowrk_model': self.netowrk_model,
            'number_of_nodes': self.G.number_of_nodes(),
            'number_of_edges': self.G.number_of_edges(),
            'density': self.density,
            'diameter': self.diameter,
            'radius': self.radius,
            'gamma': self.gamma,
            'beta': self.beta,
            'tmax': self.tmax,
            'R0': self.R_0,
            'simulation_type': self.simulation_type,
            'average_degree': self.average_degree,
            'average_shortest_path': self.average_shortest_path,
            'is_real_network': self.is_real_network
        })

    def run_stats(self, betweenness, closeness, degree, eigenvector, clustering, _is):
        return [{
                'node': i,
                'degree_centrality': round(degree[i], self.round_len),
                'closeness_centrality': round(closeness[i], self.round_len),
                'betweenness_centrality': round(betweenness[i], self.round_len),
                'eigenvector_centrality': round(eigenvector[i], self.round_len),
                'clustering_coefficient': round(clustering[i], self.round_len),
                } for i in _is]

    def statistics(self, sim):

        _is = self.get_initial_infecteds(sim)
        print(f'Nós iniciais infectados: {_is}')

        print("\n%40s    %s" % ('Process', 'Time'))
        print("%40s    %s" % ('='*40, '='*25))

        stats = None

        if _is != None:
            if self.calculated_statistics == None:
                # Run betweenness centrality
                betweenness_dict = self.stats.betweenness_centrality(self.G)
                # Run closeness centrality
                closeness_dict = self.stats.closeness_centrality(self.G, _is)
                # Run degree centrality
                degree_dict = self.stats.degree_centrality(self.G)
                # Run eigenvector centrality
                eigenvector_dict = self.stats.eigenvector_centrality(self.G)
                # Run clustering coefficient
                clustering_dict = self.stats.clustering_coefficient(self.G)

            if self.is_real_network == True:
                if self.calculated_statistics is None:
                    self.calculated_statistics = self.run_stats(
                        betweenness_dict, closeness_dict, degree_dict, eigenvector_dict, clustering_dict, _is)
                    stats = self.calculated_statistics
                else:
                    stats = None
            else:
                stats = self.run_stats(
                    betweenness_dict, closeness_dict, degree_dict, eigenvector_dict, clustering_dict, _is)

        self.calculated_statistics = None
        self.start_time = time()

        element, tmax = self.get_tmax(self.summary)

        db = DataBase()
        db.insert_simulation({
            '_id': str(uuid.uuid4()),
            'simulation_name': str(self.simulation_name),
            'iteration_number': self.iteration_number,
            'max_infection_time': tmax,
            'max_infection_len': element,
            'T0': {
                'S': int(self.summary[1]['S'][0]),
                'I': int(self.summary[1]['I'][0])
            },
            'TMAX': {
                'S': int(self.summary[1]['S'][-1:][0]),
                'I': int(self.summary[1]['I'][-1:][0]),
                'R': int(self.summary[1]['R'][-1:][0])
            },
            'stats': stats
        })
        self.log_with_time('Salvando no banco de dados', False)

    def get_tmax(self, sim):
        lst = list(sim[1]['I'])
        element = int(max(lst))
        return element, lst.index(element)

    def generate_gexf(self, sim):
        try:
            self.start_time = time()
            status = sim.get_statuses(time=self.tmax)
            last_iteration = {i: status[i]
                              for i in range(self.G.number_of_nodes())}
            self.log_with_time(
                'Obtém as informações da última iteração', False)

            self.start_time = time()

            pos = nx.spring_layout(self.G, dim=4, scale=1000)

            for node in self.G.nodes:
                status = last_iteration[node]

                self.G.nodes[node]['label'] = f'{node}_{status}'
                self.G.nodes[node]['viz'] = {'size': 10}

                if status == 'S':
                    self.G.nodes[node]['viz']['color'] = {
                        'a': 0.5, 'r': 0, 'g': 100, 'b': 0}
                elif status == 'I':
                    self.G.nodes[node]['viz']['color'] = {
                        'a': 1, 'r': 255, 'g': 0, 'b': 0}
                else:
                    self.G.nodes[node]['viz']['color'] = {
                        'a': 0.15, 'r': 100, 'g': 100, 'b': 100}

                self.G.nodes[node]['viz']['position'] = {
                    'x': pos[node][0], 'y': pos[node][1], 'z': 5}

            nx.write_gexf(self.G, f'{self.path}/network.gexf')
            self.log_with_time('Successfully generated GEXF file', False)
        except:
            pass

    def rate_function(self, G, node, status, parameters):
        tau, gamma = parameters
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

    def simulation_in_graph(self):
        print('Iniciando simulação EoN')

        sim = None

        if self.simulation_type == 'fast_sir':
            #R_0 = self.I_0/self.population
            sim = EoN.fast_SIR(self.G,
                               tau=self.beta,
                               gamma=self.gamma,
                               rho=self.R_0,
                               # transmission_weight="weight",
                               return_full_data=True,
                               tmax=self.tmax)
        elif self.simulation_type == 'complex_contagion':
            IC = defaultdict(lambda: 'S')
            for _ in range(2):
                IC[random.choice(list(self.G.nodes))] = 'I'

            sim = EoN.Gillespie_complex_contagion(
                self.G,
                self.rate_function,
                self.transition_choice,
                self.get_influence_set,
                IC,
                return_statuses=('S', 'I', 'R'),
                parameters=(self.beta, self.gamma),
                return_full_data=True,
                tmax=self.tmax)

        print('Simulação EoN finalizada')

        self.summary = sim.summary()
        self.save_sir()

        #print('Gerando arquivos')
        # self.generate_files(sim)

        print('Gerando as estatísticas da rede')

        self.statistics(sim)
        self.generate_gexf(sim)


class GraphGenerator():

    def create_net_erdos_renyi(self, nodes, probability):
        self.netowrk_model = 'erdos_renyi_gnp'
        G = nx.gnp_random_graph(nodes, probability)
        isolates = list(nx.isolates(G))
        if len(isolates) > 0:
            return G.remove_nodes_from(list(nx.isolates(G)))
        else:
            return G

    def create_net_barabasi_albert(self, nodes, edges):
        self.netowrk_model = 'barabasi_albert'
        G = nx.barabasi_albert_graph(nodes, edges)
        isolates = list(nx.isolates(G))
        if len(isolates) > 0:
            return G.remove_nodes_from(list(nx.isolates(G)))
        else:
            return G

    def create_net_watts_strogatz(self, nodes, edges, probability):
        self.netowrk_model = 'watts_strogatz'
        G = nx.watts_strogatz_graph(nodes, edges, probability)
        isolates = list(nx.isolates(G))
        if len(isolates) > 0:
            return G.remove_nodes_from(list(nx.isolates(G)))
        else:
            return G

    def create_net_facebook(self, model_id):
        self.is_real_network = True
        if model_id == 'Ego':
            self.netowrk_model = 'snap-Ego'
            # return nx.read_edgelist("../data/facebook_combined.txt.gz", create_using=nx.Graph(), nodetype=int)
            return nx.read_gexf("../data/facebook_combined.gexf")
        elif model_id == 'Bowdoin47':
            self.netowrk_model = 'socfb-Bowdoin47'
            # return nx.read_adjlist("../data/socfb-Bowdoin47.mtx", create_using=nx.DiGraph(), nodetype=int)
            return nx.read_gexf("../data/socfb-Bowdoin47.gexf")
        elif model_id == 'Haverford76':
            self.netowrk_model = 'socfb-Haverford76'
            # return nx.read_adjlist("../data/socfb-Haverford76.mtx", create_using=nx.DiGraph(), nodetype=int)
            return nx.read_gexf("../data/socfb-Haverford76.gexf")
        elif model_id == 'Simmons81':
            self.netowrk_model = 'socfb-Simmons81'
            # return nx.read_adjlist("../data/socfb-Simmons81.mtx", create_using=nx.DiGraph(), nodetype=int)
            return nx.read_gexf("../data/socfb-Simmons81.gexf")
        elif model_id == 'Facebook01':
            self.netowrk_model = 'fb-01'
            return nx.read_gexf('../data/fb1.gexf')
        elif model_id == 'Facebook02':
            self.netowrk_model = 'fb-02'
            return nx.read_gexf('../data/fb2.gexf')
        elif model_id == 'Facebook03':
            self.netowrk_model = 'fb-03'
            return nx.read_gexf('../data/fb3.gexf')
        elif model_id == 'Facebook04':
            self.netowrk_model = 'fb-04'
            return nx.read_gexf('../data/fb4.gexf')
        else:
            None

    def create_net_twitter(self):
        self.is_real_network = True
        self.netowrk_model = 'twitter'
        return nx.read_edgelist("../data/twitter_combined.txt.gz", create_using=nx.Graph(), nodetype=int)
