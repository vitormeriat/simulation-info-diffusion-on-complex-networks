import net_sir_sim as sim
import enum


class EnumResult():
    def get(self, msg):
        return msg.value


@enum.unique
class BaseModel(enum.Enum):
    EGO = 'Ego'
    BOWDOIN = 'Bowdoin47'
    HAVERFORD = 'Haverford76'
    SIMMONS = 'Simmons81'
    FB01 = 'Facebook01'
    FB02 = 'Facebook02'
    FB03 = 'Facebook03'
    FB04 = 'Facebook04'
    TWITTER = 'Twitter'


@enum.unique
class SimulationType(enum.Enum):
    FAST_SIR = 'fast_sir'
    COMPLEX_CONTAGION = 'complex_contagion'


def simule(gamma, beta, R_0, tmax, simulation_name, num_simulations, model, base_model, param, simulation_type):
    graph = sim.Graph()

    if model == 'facebook':
        G = graph.create_net_facebook(model_id=base_model)
    elif model == 'twitter':
        G = graph.create_net_twitter()
    elif model == 'watts_strogatz':
        G = graph.create_net_watts_strogatz(
            nodes=param['watts_strogatz']['nodes'],
            edges=param['watts_strogatz']['edges'],
            probability=param['watts_strogatz']['probability'])
    elif model == 'barabasi_albert':
        G = graph.create_net_barabasi_albert(
            nodes=param['barabasi_albert']['nodes'],
            edges=param['barabasi_albert']['edges'])
    elif model == 'erdos_renyi':
        G = graph.create_net_erdos_renyi(
            nodes=param['erdos_renyi']['nodes'],
            probability=param['erdos_renyi']['probability'])

    graph.configuration(G, gamma, beta, R_0, tmax, simulation_type)
    simulation = sim.Simulation(num_simulations, simulation_name)
    if simulation.initialize(graph):
        simulation.simulate()


def generate_sim(base_model, gamma, beta, R_0, tmax, simulation_name, num_simulations, simulation_type):
    base = sim.Base()

    if base.query_yes_no("\nDeseja prosseguir na simulação?", None):
        ntwrks = ["Ego", "Bowdoin47", "Haverford76", "Simmons81",
                  "Facebook01", "Facebook02", "Facebook03", "Facebook04", "Twitter"]
        parametrs = [
            {
                'model': 'Ego',
                'watts_strogatz': {'nodes': 4039, 'edges': 45, 'probability': 0.4},
                'barabasi_albert': {'nodes': 4039, 'edges': 22},
                'erdos_renyi': {'nodes': 4039, 'probability': 0.01083}
            },
            {
                'model': 'Bowdoin47',
                'watts_strogatz': {'nodes': 2252, 'edges': 75, 'probability': 0.4},
                'barabasi_albert': {'nodes': 2252, 'edges': 37},
                'erdos_renyi': {'nodes': 2252, 'probability': 0.0331}
            },
            {
                'model': 'Haverford76',
                'watts_strogatz': {'nodes': 1446, 'edges': 83, 'probability': 0.4},
                'barabasi_albert': {'nodes': 1446, 'edges': 42},
                'erdos_renyi': {'nodes': 1446, 'probability': 0.05683}
            },
            {
                'model': 'Simmons81',
                'watts_strogatz': {'nodes': 1518, 'edges': 44, 'probability': 0.4},
                'barabasi_albert': {'nodes': 1518, 'edges': 22},
                'erdos_renyi': {'nodes': 1518, 'probability': 0.02883}
            },
            {
                'model': 'Facebook01',
                'watts_strogatz': {'nodes': 168, 'edges': 2, 'probability': 0.4},
                'barabasi_albert': {'nodes': 168, 'edges': 1},
                'erdos_renyi': {'nodes': 168, 'probability': 0.012}
            },
            {
                'model': 'Facebook02',
                'watts_strogatz': {'nodes': 333, 'edges': 32, 'probability': 0.4},
                'barabasi_albert': {'nodes': 333, 'edges': 16},
                'erdos_renyi': {'nodes': 333, 'probability': 0.09}
            },
            {
                'model': 'Facebook03',
                'watts_strogatz': {'nodes': 224, 'edges': 2, 'probability': 0.4},
                'barabasi_albert': {'nodes': 224, 'edges': 1},
                'erdos_renyi': {'nodes': 224, 'probability': 0.009}
            },
            {
                'model': 'Facebook04',
                'watts_strogatz': {'nodes': 59, 'edges': 4, 'probability': 0.4},
                'barabasi_albert': {'nodes': 59, 'edges': 2},
                'erdos_renyi': {'nodes': 59, 'probability': 0.058}
            },
            {
                'model': 'Twitter',
                'watts_strogatz': {'nodes': 81306, 'edges': 34, 'probability': 0.4},
                'barabasi_albert': {'nodes': 81306, 'edges': 17},
                'erdos_renyi': {'nodes': 81306, 'probability': 0.00044}
            }
        ]

        idx = ntwrks.index(base_model)
        param = parametrs[idx]

        simulation_name = f'{simulation_name}-{base_model}'

        if base_model == 'Twitter':
            simule(gamma, beta, R_0, tmax, f'{simulation_name}-twitter',
                   num_simulations, 'twitter', base_model, param, simulation_type)
        else:
            simule(gamma, beta, R_0, tmax, f'{simulation_name}-facebook',
                   num_simulations, 'facebook', base_model, param, simulation_type)

        #simule(gamma, beta, R_0, tmax, f'{simulation_name}-facebook', num_simulations, 'facebook', base_model, param, simulation_type)

        simule(gamma, beta, R_0, tmax, f'{simulation_name}-watts_strogatz',
               num_simulations, 'watts_strogatz', base_model, param, simulation_type)

        simule(gamma, beta, R_0, tmax, f'{simulation_name}-barabasi_albert',
               num_simulations, 'barabasi_albert', base_model, param, simulation_type)

        simule(gamma, beta, R_0, tmax, f'{simulation_name}-erdos_renyi',
               num_simulations, 'erdos_renyi', base_model, param, simulation_type)
    else:
        print('Simulação abortada!!!')
