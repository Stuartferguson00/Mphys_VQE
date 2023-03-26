import pickle
from Solver_folder.Variational_Algorithm_ import Variational_Algorithm as VA
import matplotlib.pyplot as plt
from mitiq.zne.scaling.folding import fold_gates_at_random
from qiskit_aer.backends import QasmSimulator
plt.style.use('classic')
from Solver_folder.Circuit_Builder import *




def evaluate_graph(size_graph, num_ansatz_layers, noise_dict, G,  ZNE, CDR, vnCDR, rand_params = None):
    """

    Function that evaluates a hardware efficient ansatz, given variational parameters corresponding to a minima
    or can create own random variational parameters.

    Parameters
    ----------
    size_graph : int
        how many nodes on graph (ie. how many qubits will be required

    num_ansatz_layers : int
        how many layers of rotation gates in ansatz

    noise_dict : dict
        dictionary detailing noise model. It is standard throughout the code

    G : obj
        networkx.classes.graph.Graph object, ie. the max-cut problem

    ZNE : dict
        dictionary detailing ZNE mitigation. It is standard throughout the code

    CDR : dict
        dictionary detailing CDR mitigation. It is standard throughout the code

    vnCDR : dict
        dictionary detailing vnCDR mitigation. It is standard throughout the code

    rand_params : array
        array of length size_graph * num_ansatz_layers, with required variational parameters


    Returns
    -------

    """



    circuit = hardware_efficient_build(size_graph, num_ansatz_layers)
    if type(rand_params) != type(np.array([1,1])):
        rand_params = np.random.uniform(0, 2 * pi, len(circuit.parameters))
    else:
        #check manual rand params is the correct length:
        if len(circuit.parameters) != len(rand_params):
            raise Exception('rand_params must be of length size_graph * num_ansatz_layers')



    seed = 10598
    #Define prefered backends for simulation both noisy and idea;=l
    backend = QasmSimulator()
    sim_inst = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
    backend = QasmSimulator(method="density_matrix")
    noisy_inst = QuantumInstance(backend, seed_simulator=seed,
                                 seed_transpiler=seed,
                                 noise_model=noise_dict["noise_model"],
                                 coupling_map=noise_dict["coupling_map"],
                                 basis_gates=noise_dict["basis_gates"],
                                 optimization_level=0)

    #initialise VA clas sinstance
    VQE = VA(circuit, noisy_inst, sim_inst, G, noise_dict=noise_dict)

    #run the circuit, with optional mitigation dependant on ZNE, CDR and vnCDR params
    all_results, all_errs, all_names = VQE.mitigate_random_circuit(rand_params, num_shots=8182,  ZNE=ZNE, CDR=CDR, vnCDR=vnCDR ,  _MCMC=True)



    return all_results, all_errs, circuit




def main():

    """

    main function to run a max-cut VQE, where the many parameters can be adjusted
    Intended for use in experiment, so to parameters are altered manually within this function.
    outputs saved to file in circuit_results

    """



    #define max-cut hyper-parameters
    size = 6
    num_reps = 1
    num_ansatz_layers = 4

    #define mitigation hyper parameters
    num_training_circuits = 8
    fraction_non_clifford = 0.25
    scale_factors = (1, 2, 3)

    #define depolarising error rates
    #one_qubit_err_rate = 0.00039
    #two_qubit_err_rate = 0.018
    one_qubit_err_rate = 0.0004
    two_qubit_err_rate = 0.02
    dr = np.asarray([one_qubit_err_rate, one_qubit_err_rate, one_qubit_err_rate, two_qubit_err_rate, two_qubit_err_rate, 0])
    NM, BG = depolarising_NM(dr[0], dr[1], dr[2], dr[3], dr[4], dr[5])

    #define crosstalk noise model
    #idle_ZZ and driven_ZZ refer to rotation required by each gate each layer of the circuit
    #driven is currently untested
    CM = CouplingMap().from_ring(size)
    noise_dict = {"idle_zz" : 0.01, "driven_zz" : 0,
                  "noise_model": NM, "basis_gates": BG, "coupling_map": CM}

    # ZNE_during_mini = {"scale_factors": scale_factors, "scale_noise": fold_gates_at_random }
    ZNE_during_mini = None

    #define required mitigation dicts
    #comment out each method depending on what you require
    #ZNE = {"scale_factors": scale_factors, "scale_noise": fold_gates_at_random}
    ZNE = None
    #CDR= {"num_training_circuits" : num_training_circuits, "fraction_non_clifford" : fraction_non_clifford }
    CDE = None
    #vnCDR =  {"scale_factors": scale_factors, "scale_noise": fold_gates_at_random , "num_training_circuits" : num_training_circuits, "fraction_non_clifford" : fraction_non_clifford }
    vnCDR = None



    #load in previously dictated graph(s)
    imp_Graphs = pickle.load(open("Graphs/Graphs_"+ str(size)+ ".pkl", 'rb'))

    #load in minimised circuit parameters
    #optionally make this = None if random parameters required
    min_params = np.loadtxt("opt_params/x_" + str(size), delimiter=',')
    min_params = [None] * num_reps


    #loop through graphs and random/minimised parameters as required
    n = size
    all_results = []
    all_errs = []
    print("starting " + str(size))
    for i in range(num_reps):
        print("This is repetition number " + str(i))
        print(i)

        #evaluate the graph at required variational parameters, and mitigate
        results, errs, circ = evaluate_graph(size, num_ansatz_layers, noise_dict, imp_Graphs[i], ZNE, CDR,vnCDR, rand_params =min_params[i])
        all_results.append(results)
        all_errs.append(errs)
    print("done " + str(size))

    #save results as numpy arrays, where rows are labelled ['Noisy', 'ZNE', 'CDR', 'vnCDR', 'ideal'], assuming all mitigation method are required
    np.savetxt("circuit_results/rand_ZZ_errs_"+ str(size), all_errs, delimiter=',')
    np.savetxt("circuit_results/rand_ZZ_res_"+ str(size), all_results, delimiter=',')





