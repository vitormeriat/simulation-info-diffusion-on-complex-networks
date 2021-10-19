import net_sir_sim as sim


def simule(G):
    graph = sim.Graph()
    graph.configuration(G, gamma, beta, R_0, tmax)
    simulation = sim.Simulation(simulation_name, num_simulations)
    if simulation.initialize(graph):
        simulation.simulate()


def generate_sim(base_model="", gamma, beta, R_0, tmax):
    ntwrks = ["Ego", "Bowdoin47", "Haverford76", "Simmons81"]
    parametrs = [
        {
            'model': 'Ego',
            'watts_strogatz': {'nodes'= 4039, 'edges' = 45, 'probability' = 0.4},
            'barabasi_albert': {'nodes' = 4039, 'edges' = 22},
            'erdos_renyi': {'nodes'= 4039, 'probability' = 0.01083}
        },
        {
            'model': 'Bowdoin47',
            'watts_strogatz': {'nodes'= 2252, 'edges' = 75, 'probability' = 0.4},
            'barabasi_albert': {'nodes' = 2252, 'edges' = 37},
            'erdos_renyi': {'nodes'= 2252, 'probability' = 0.0331}
        },
        {
            'model': 'Haverford76',
            'watts_strogatz': {'nodes'= 1446, 'edges' = 83, 'probability' = 0.4},
            'barabasi_albert': {'nodes' = 1446, 'edges' = 42},
            'erdos_renyi': {'nodes'= 1446, 'probability' = 0.05683}
        },
        {
            'model': 'Simmons81',
            'watts_strogatz': {'nodes'= 1518, 'edges' = 44, 'probability' = 0.4},
            'barabasi_albert': {'nodes' = 1518, 'edges' = 22},
            'erdos_renyi': {'nodes'= 1518, 'probability' = 0.02883}
        },
    ]
    idx = ntwrks.index(base_model)
    param = parametrs[idx]

    G = graph.create_net_facebook(model_id=base_model)
    simule(G, gamma, beta, R_0, tmax)

    G = graph.create_net_watts_strogatz(
        nodes=param['watts_strogatz']['nodes'],
        edges=param['watts_strogatz']['edges'],
        probability=param['watts_strogatz']['probability'])
    simule(G, gamma, beta, R_0, tmax)

    G = graph.create_net_barabasi_albert(
        nodes=param['barabasi_albert']['nodes'],
        edges=param['barabasi_albert']['edges'])
    simule(G, gamma, beta, R_0, tmax)

    G = graph.create_net_erdos_renyi(
        nodes=param['erdos_renyi']['nodes'],
        probability=param['erdos_renyi']['probability'])
    simule(G, gamma, beta, R_0, tmax)


if __name__ == "__main__":
    graph = sim.Graph()
    base = sim.Base()

    # =============== PARAMETERIZATION =======================================

    simulation_name = 'teste-xpto'
    num_simulations = 5
    gamma = 0.3  # 1.0
    beta = 0.109
    R_0 = None
    tmax = 10

    #G = graph.create_net_erdos_renyi(nodes=4039, probability=0.0109)
    #G = graph.create_net_barabasi_albert(nodes=4039, edges=22)
    #G = graph.create_net_watts_strogatz(nodes=4039, edges=45, probability=0.4)
    #G = graph.create_net_facebook(model_id='01')

    # ========================================================================

    if G != None:
        graph.configuration(G, gamma, beta, R_0, tmax)

        if base.query_yes_no("\nDeseja prosseguir na simulação?", None):
            simulation = sim.Simulation(
                simulation_name=simulation_name,
                num_simulations=num_simulations)
            if simulation.initialize(graph):
                simulation.simulate()
        else:
            print('Simulação abortada!!!')
    else:
        print('Não foi possível gerar esta rede')
