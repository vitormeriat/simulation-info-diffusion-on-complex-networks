import enum
import utils as utl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--foo', help='foo help')
args = parser.parse_args()

if __name__ == "__main__":
    config = utl.EnumResult()

    # =============== PARAMETERIZATION =======================================

    simulation_name = 'Teste2-B-EGO-COMPLEX-S4'#'FB001-simples-03'#'Simmons-Complex-f10'
    num_simulations = 1
    gamma = 0.4
    beta = 0.3
    tmax = 30
    R_0 = None

    # Ego - Bowdoin47 - Haverford76 - Simmons81 - FB01 - FB02 - FB03 - FB04
    facebook_base_model = config.get(utl.BaseModel.TWITTER)

    # fast_sir - complex_contagion
    simulation_type = config.get(utl.SimulationType.COMPLEX_CONTAGION)

    # ========================================================================

    utl.generate_sim(facebook_base_model, gamma, beta, R_0, tmax, simulation_name, num_simulations, simulation_type)
