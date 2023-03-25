import sys
import qiskit_aer

import mitiq


#sys.path.append("/home/jovyan/Max_Cut/ZZ_NOISE_FIX/Solver_folder")
import pickle
from Solver_folder.Variational_Algorithm_ import Variational_Algorithm as VA

import qiskit_aer
import smtplib
import sys
from scipy.optimize import curve_fit
import numpy
import time

import statsmodels as sm

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
import networkx as nx




from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
import networkx as nx


import matplotlib.pyplot as plt
from qiskit.opflow import DictStateFn
from qiskit.compiler import transpile
import copy
import mitiq
import scipy
from mitiq.zne import mitigate_executor
from mitiq.zne.scaling.folding import fold_gates_at_random
import logging
from time import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import qiskit
import numpy as np
from functools import partial
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import (
    CircuitSampler,
    CircuitStateFn,
    ExpectationBase,
    ExpectationFactory,
    ListOp,
    OperatorBase,
    PauliSumOp,
    StateFn,
    OperatorStateFn
)
from qiskit.opflow.gradients import GradientBase
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.utils.backend_utils import is_aer_provider
from qiskit.utils.validation import validate_min

from qiskit.algorithms.aux_ops_evaluator import eval_observables
from qiskit.algorithms.exceptions import AlgorithmError
from qiskit.algorithms.list_or_dict import ListOrDict
from qiskit.algorithms.optimizers import SLSQP, Minimizer, Optimizer
from qiskit.algorithms.variational_algorithm import VariationalAlgorithm, VariationalResult
from qiskit.algorithms.minimum_eigen_solvers.minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult

#from mitiq.zne.scaling.folding import fold_gates_from_right

from mitiq import zne
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import qiskit.providers.aer.noise as noise
from qiskit import IBMQ, Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from ibm_quantum_widgets import CircuitComposer
import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import QuantumCircuit, Parameter
from numpy import pi

from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms import VQE, NumPyMinimumEigensolver, QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit.transpiler import CouplingMap
from collections import OrderedDict
from qiskit.circuit import ParameterVector, QuantumCircuit






from qiskit_aer.backends import QasmSimulator


import time
plt.style.use('classic')
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.tools.visualization import dag_drawer
from qiskit.circuit.library import CHGate, U2Gate, CXGate, RZZGate
from qiskit.converters import dag_to_circuit

from Solver_folder.Circuit_Builder import *


#provider = IBMQ.enable_account("e810c004e1eceb26ccb71869ec3f769d7bfb9f8c9b52a0e9407ecaeba88a95fa181a4e187c4a8599dc0d6cb426181b68d43980342851973c747f9bdcad35224a")


def evaluate_random_graph(size_graph, num_ansatz_layers, noise_dict, G,  ZNE, CDR, vnCDR, rand_params = None):
    n = size_graph
    # G= nx.erdos_renyi_graph(size_graph,0.5)

    circuit = hardware_efficient_build(size_graph, num_ansatz_layers)
    if type(rand_params) != type(np.array([1,1])):
        rand_params = np.random.uniform(0, 2 * pi, len(circuit.parameters))
    else:
        #check manual rand params is the correct length:
        if len(circuit.parameters) != len(rand_params):
            print("not workinggg oh nooo")
    #CM = CouplingMap().from_hexagonal_lattice(3, 3)



    seed = 10598
    # cant really run anything other than the basic Qasm.. it doesn tlike extended simulator although thats 100% what its doing...
    backend = QasmSimulator()
    sim_inst = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

    backend = QasmSimulator(method="density_matrix")
    noisy_inst = QuantumInstance(backend, seed_simulator=seed,
                                 seed_transpiler=seed,
                                 noise_model=noise_dict["noise_model"],
                                 coupling_map=noise_dict["coupling_map"],
                                 basis_gates=noise_dict["basis_gates"],
                                 optimization_level=0)

    VQE = VA(circuit, noisy_inst, sim_inst, G, noise_dict=noise_dict)

    all_results, all_errs, all_names = VQE.mitigate_random_circuit(rand_params, num_shots=1000,  ZNE=ZNE, CDR=CDR, vnCDR=vnCDR ,  _MCMC=True)



    return all_results, all_errs, circuit






sizes = [6]

num_reps = 20
num_ansatz_layers = 4
num_training_circuits = 80
fraction_non_clifford = 0.5
scale_factors = (1, 2, 3)

#dr = np.asarray([0.0002,0.0002,0.0002,0.01,0.01,0])
one_qubit_err_rate = 0.00037
two_qubit_err_rate = 0.018
#one_qubit_err_rate = 0.0004
#two_qubit_err_rate = 0.02
dr = np.asarray([one_qubit_err_rate, one_qubit_err_rate, one_qubit_err_rate, two_qubit_err_rate, two_qubit_err_rate, 0])
NM, BG = depolarising_NM(dr[0], dr[1], dr[2], dr[3], dr[4], dr[5])
#CM = CouplingMap().from_hexagonal_lattice(3, 3)
CM = CouplingMap().from_grid(3, 3)


noise_dict = {"idle_zz" : 0.01, "driven_zz" : 0,
#noise_dict = {"idle_zz": 0, "driven_zz": 0,
              "noise_model": NM, "basis_gates": BG, "coupling_map": CM}

# ZNE_during_mini = {"scale_factors": scale_factors, "scale_noise": fold_gates_at_random }
ZNE_during_mini = None

ZNE = {"scale_factors": scale_factors, "scale_noise": fold_gates_at_random}
#ZNE = None


CDR= {"num_training_circuits" : num_training_circuits, "fraction_non_clifford" : fraction_non_clifford }
#CDR = None

vnCDR =  {"scale_factors": scale_factors, "scale_noise": fold_gates_at_random , "num_training_circuits" : num_training_circuits, "fraction_non_clifford" : fraction_non_clifford }
#vnCDR = None




all_all_results = []
all_all_errs = []


for size in sizes:
    imp_Graphs = pickle.load(open("Graphs_"+ str(size)+ ".pkl", 'rb'))
    min_params = np.loadtxt("VQE_results/VQE_ZZ_ZNE/x_" + str(size), delimiter=',')
    min_params_2 = np.loadtxt("VQE_results/VQE_ZZ_ZNE/r_x_" + str(size), delimiter=',')
    min_params = np.vstack((min_params, min_params_2))


    n = size
    all_results = []
    all_errs = []
    print("starting " + str(size))
    for i in range(num_reps):
        print(i)
        results, errs, circ = evaluate_random_graph(size, num_ansatz_layers, noise_dict, imp_Graphs[i], ZNE, CDR,vnCDR)#, rand_params =_params[i])
        all_results.append(results)
        all_errs.append(errs)
    print("done " + str(size))

    np.savetxt("rc_results_2/rand_ZZ_errs_"+ str(size), all_errs, delimiter=',')
    np.savetxt("rc_results_2/rand_ZZ_res_"+ str(size), all_results, delimiter=',')





