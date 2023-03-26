import pickle
from Solver_folder.Variational_Algorithm_ import Variational_Algorithm as VA
import matplotlib.pyplot as plt
from mitiq.zne.scaling.folding import fold_gates_at_random
from qiskit_aer.backends import QasmSimulator
plt.style.use('classic')
from Solver_folder.Circuit_Builder import *

def find_avg(noise_dict, num_evals = 10, max_iters= 100,
                    size_graph = 5,
                    num_ansatz_layers = 4,
                    graphs =None,
                    ZNE = None,
                    CDR = None,
                    vnCDR = None,
                    ZNE_during_mini = None
                    ):

    """

    Function to run multiple max-cut problems with a variety of mitigation options


    Parameters
    ----------
    noise_dict : dict
        dictionary detailing noise model. It is standard throughout the code

    num_evals : int
        Number of max-cuts to evaluate

    max_iters : int
        Number of minimisation steps to proceed

    size_graph : int
        how many nodes on graph (ie. how many qubits will be required

    num_ansatz_layers : int
        how many layers of rotation gates in ansatz

    Graphs : obj
        networkx.classes.graph.Graph object, ie. the max-cut problem

    ZNE : dict
        dictionary detailing ZNE mitigation post minimisation. It is standard throughout the code

    CDR : dict
        dictionary detailing CDR mitigation post minimisation. It is standard throughout the code

    vnCDR : dict
        dictionary detailing vnCDR mitigation post minimisation. It is standard throughout the code

    ZNE_during_mini : dict
        dictionary detailing ZNE for use throughout mitigation. It is standard throughout the code

    Returns
    -------

    """


    # provider = IBMQ.load_account()
    start = time.time()


    vqe_results = []
    vqe_errs  = []
    offsets = []
    #opt params:
    x_ = []


    for l in range(num_evals):
        print("this is the " + str(l) + "th average loop")
        if graphs != None:
            G = graphs[l]
        else:
            G = nx.erdos_renyi_graph(size_graph, 0.5)

        circuit = hardware_efficient_build(size_graph, num_ansatz_layers)


        seed = 10598
        # cant really run anything other than the basic Qasm.. it doesn tlike extended simulator although thats 100% what its doing...
        backend = QasmSimulator()
        sim_inst = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

        backend = QasmSimulator(method="density_matrix")
        noisy_inst = QuantumInstance(backend, seed_simulator=seed,
                                     seed_transpiler=seed,
                                     noise_model= noise_dict["noise_model"],
                                     coupling_map= noise_dict["coupling_map"],
                                     basis_gates= noise_dict["basis_gates"],
                                     optimization_level = 0)

        VQE = VA(circuit, noisy_inst, sim_inst, G, noise_dict=noise_dict)


        # vqe_result, vqe_err= miti.run(CDR = True, scale_factors = scale_factors ,num_training_circuits = num_training_circuits, comparisons = True, ZNE_during_mini = True)
        all_results, all_errs, all_names= VQE.run(maxiters = max_iters,
                                                    num_shots = 8192,
                                                    ZNE = ZNE,
                                                    CDR = CDR,
                                                    vnCDR = vnCDR,
                                                    ZNE_during_mini = ZNE_during_mini)
                                        # ZNE_during_mini=ZNE_during_mini)

        x_.append(VQE.opt_params)

        vqe_results.append(all_results)
        vqe_errs.append(all_errs)
        offsets.append(VQE.offset)

    end = time.time()
    print("total time spent: " + str(end - start))

    return vqe_results, vqe_errs, offsets, all_names


def main():

    """

    main function to run many max-cut VQE, where the many parameters can be adjusted
    Intended for use in experiment, so to parameters are altered manually within this function.
    outputs saved to file in VQE_results

    """

    circ_reps = 1

    max_iters = 5
    num_training_circuits =4
    fraction_non_clifford = 0.5
    scale_factors = (1, 2, 3)
    num_ansatz_layers = 4

    size = 6
    print("starting " + str(size))

    #ectract graphs to be evaluated
    graphs = pickle.load(open("Graphs/Graphs_"+str(size)+".pkl", 'rb'))
    print(len(graphs))
    repeat_graphs = []
    for i in range(circ_reps):
        repeat_graphs.append(graphs[1])





    # define depolarising error rates
    # one_qubit_err_rate = 0.00039
    # two_qubit_err_rate = 0.018
    one_qubit_err_rate = 0.0004
    two_qubit_err_rate = 0.02
    dr = np.asarray(
        [one_qubit_err_rate, one_qubit_err_rate, one_qubit_err_rate, two_qubit_err_rate, two_qubit_err_rate, 0])
    NM, BG = depolarising_NM(dr[0], dr[1], dr[2], dr[3], dr[4], dr[5])
    # define crosstalk noise model
    # idle_ZZ and driven_ZZ refer to rotation required by each gate each layer of the circuit
    # driven is currently untested
    CM = CouplingMap().from_ring(size)
    noise_dict = {"idle_zz": 0.01, "driven_zz": 0,
                  "noise_model": NM, "basis_gates": BG, "coupling_map": CM}

    #ZNE_during_mini = {"scale_factors": scale_factors, "scale_noise": fold_gates_at_random }
    ZNE_during_mini = None

    # define required mitigation dicts
    #comment out each method depending on what you require
    #ZNE = {"scale_factors": scale_factors, "scale_noise": fold_gates_at_random }
    ZNE = None
    #CDR = {"num_training_circuits" : num_training_circuits, "fraction_non_clifford" : fraction_non_clifford }
    CDR = None
    #vnCDR =  {"scale_factors": scale_factors, "scale_noise": fold_gates_at_random , "num_training_circuits" : num_training_circuits, "fraction_non_clifford" : fraction_non_clifford }
    vnCDR = None




    VQE_results, VQE_errs, offsets, all_names = find_avg(noise_dict,
                                            num_evals = circ_reps,
                                            max_iters= max_iters,
                                            size_graph = size,
                                            num_ansatz_layers = num_ansatz_layers,
                                            graphs = repeat_graphs,
                                            ZNE = ZNE,
                                            CDR = CDR,
                                            vnCDR = vnCDR,
                                            ZNE_during_mini = ZNE_during_mini)




    VQE_errs = np.asarray(VQE_errs)
    VQE_results = np.asarray(VQE_results)
    offsets = np.asarray(offsets)

    # save results as numpy arrays, where rows are labelled ['Noisy', 'ZNE', 'CDR', 'vnCDR','ideal', 'actual minima'], assuming all mitigation methods are selected
    np.savetxt("VQE_results/results_"+str(size),VQE_results , delimiter=',')
    np.savetxt("VQE_results/errs"+str(size),VQE_errs , delimiter=',')
    np.savetxt("VQE_results/offsets"+str(size),offsets , delimiter=',')

    return

main()





