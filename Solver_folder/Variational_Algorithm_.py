from mitiq.zne.scaling import fold_gates_at_random
from .Circuit_Builder import *
from .Miti_Techniques import execute_with_cdr_alt, mock_execute_with_zne
from qiskit.algorithms import VQE, NumPyMinimumEigensolver

from qiskit.transpiler import CouplingMap

from qiskit_optimization.applications import Maxcut
import time


class Variational_Algorithm:

    def __init__(self, ansatz, quantum_instance, simulation_instance, G,
                 noise_dict=None
                 ):
        # others (scale_noise=fold_gates_at_random,fit_function = None, coupling_map = None)

        """
        
        Class that runs a max cut VQE


        Parameters
        ----------



        Returns
        -------
        
        
        
        """

        # define class variables

        # maybe check if this is paramatrised
        self.base_circuit = ansatz

        self.Q_inst = quantum_instance
        self.sim_inst = simulation_instance
        # define executors
        self.define_sim_ex()

        # define problem
        self.G = G

        # define size off graph hence number of qubits
        self.n = G.number_of_nodes()
        self.nfev = 0

        self.expected_result = self.find_expected_result()

        # find the expected results from G

        if noise_dict == None:
            self.noise_dict = {"idle_zz": 0.0, "driven_zz": 0,
                               "depolarising_rates": np.asarray([0.0004, 0.0004, 0.0004, 0.02, 0.02, 0]),
                               "coupling_map": CouplingMap().from_hexagonal_lattice(3, 3),
                               "basis_gates": None}
        else:
            # write something to check inserted noise dict is full of necessary information
            self.noise_dict = noise_dict

        # define empty lists necessary
        self.energys_miti = []
        self.params = []
        self.vals = []






    def find_expected_result(self):
        # classically find the expected result
        w = np.zeros([self.n, self.n])
        for i in range(self.n):
            for j in range(self.n):
                temp = self.G.get_edge_data(i, j, default=0)

                if temp != 0:
                    w[i, j] = 1  # temp["weight"]

        # map to ising hamiltonian
        max_cut = Maxcut(w)
        qp = max_cut.to_quadratic_program()
        qubitOp, offset = qp.to_ising()
        self.offset = offset
        ee = NumPyMinimumEigensolver()

        # find real result
        real_result = ee.compute_minimum_eigenvalue(qubitOp)
        real_res = real_result.eigenvalue.real
        real_res = real_res + offset
        return real_res

    def callback(self, nfev, params, val, stepsize=None, step_accept=None):
        self.params.append(params)
        self.vals.append(val)
        # print (val)

    def define_sim_ex(self):
        self.simulator = partial(self.compute_expectation, sim=True)
        self.executor = partial(self.compute_expectation, sim=False)

    def run(self, maxiters=10, num_shots=1000, ZNE_during_mini=None, ZNE=None, CDR=None, vnCDR=None):

        """
        Function to run a VQE given a circuit and a problem
        Elect which mitigation methods to use and when
        if any mitigation method does not equal None, it must be a dictionary and the mitigation method will run
        with parameters defined by the given dict
        """
        self.num_shots = num_shots
        # check wif the minimisation should be mitigated by ZNE
        if ZNE_during_mini != None:
            scale_factors = ZNE_during_mini["scale_factors"]
            scale_noise = ZNE_during_mini["scale_noise"]

            minimising_executor = partial(mock_execute_with_zne,
                                          scale_factors=scale_factors,
                                          executor=self.executor,
                                          scale_noise=scale_noise,
                                          return_err=False,
                                          base_circuit=self.base_circuit)
        else:
            minimising_executor = self.executor

        # setup minimastion
        spsa = SPSA(maxiter=maxiters, callback=self.callback)
        bounds = np.reshape([-2 * np.pi, 2 * np.pi] * len(self.base_circuit.parameters),
                            (len(self.base_circuit.parameters), 2))

        # run minimisation
        print("starting minimisation")
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

        res = spsa.minimize(minimising_executor,
                            np.random.uniform(-np.pi, np.pi, len(self.base_circuit.parameters)),
                            bounds=bounds)
        print("finished minimisation")
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

        self.res = res
        self.opt_params = res.x

        n_it = np.arange(0, len(self.vals))

        plt.xlabel("Number of minimisation iterations")
        plt.ylabel("approximate ratio of fev/exact ")
        plt.plot(n_it, self.vals, label="VQE minimisation")
        plt.plot([np.min(n_it), np.max(n_it)], [self.expected_result, self.expected_result],
                 label="Exact Answer")
        plt.show()



        if ZNE_during_mini!=None:
            ZNE = None


            scale_factors = ZNE_during_mini["scale_factors"]
            scale_noise = ZNE_during_mini["scale_noise"]

            ZNE_miti, ZNE_err = mock_execute_with_zne(self.opt_params,
                                                      scale_factors,
                                                      self.executor,
                                                      scale_noise,
                                                      base_circuit = self.base_circuit)



            all_results = [self.executor(self.opt_params), ZNE_miti ]
            all_errs = [0, ZNE_err]
            all_names = ["Noisy", "ZNE" ]  # ["Non-mitigated", "ZNE", "CDR", "vnCDR","Simulated", "Exact solution"]

        else:
            # basically, always do comparisons
            all_results = [res.fun, ]
            all_errs = [0, ]
            all_names = ["Noisy", ]  # ["Non-mitigated", "ZNE", "CDR", "vnCDR","Simulated", "Exact solution"]

        circuit = self.bind(self.opt_params)


        #Always do comparisons of different techniques available
        if ZNE != None:
            #doesnt happen if ZNE_during_miti happens
            print("Starting ZNE")
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
            scale_factors = ZNE["scale_factors"]
            scale_noise = ZNE["scale_noise"]
            ZNE_miti, ZNE_err = mock_execute_with_zne(circuit,
                                                      scale_factors,
                                                      self.executor,
                                                      scale_noise,
                                                      base_circuit = self.base_circuit)
            all_results.append(ZNE_miti)
            all_errs.append(ZNE_err)
            all_names.append("ZNE")

        if CDR != None:
            print("Starting CDR")
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
            num_training_circuits = CDR["num_training_circuits"]
            fraction_non_clifford = CDR["fraction_non_clifford"]
            CDR_miti, CDR_err = execute_with_cdr_alt(
                                    self.base_circuit,
                                    self.executor,
                                    simulator=self.simulator,
                                    seed=0,
                                    num_training_circuits=num_training_circuits,
                                    fraction_non_clifford=fraction_non_clifford,
                                    params=self.opt_params,  # if we want MCMC
                                    _MCMC=True)

            all_results.append(float(CDR_miti))
            all_errs.append(CDR_err)
            all_names.append("CDR")


        if vnCDR != None:
            print("Starting vnCDR")
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
            num_training_circuits = vnCDR["num_training_circuits"]
            fraction_non_clifford = vnCDR["fraction_non_clifford"]
            scale_factors = vnCDR["scale_factors"]
            scale_noise = vnCDR["scale_noise"]
            vnCDR_miti, vnCDR_err = execute_with_cdr_alt(
                                self.base_circuit,
                                self.executor,
                                simulator=self.simulator,
                                seed=0,
                                scale_factors=scale_factors,
                                num_training_circuits=num_training_circuits,
                                scale_noise=scale_noise,
                                fraction_non_clifford=fraction_non_clifford,
                                params=self.opt_params,  # if we want MCMC
                                _MCMC=True)
            all_results.append(vnCDR_miti)
            all_errs.append(vnCDR_err)
            all_names.append("vnCDR")

        real_energy = self.simulator(circuit)
        all_results.append(real_energy)
        all_errs.append(0)
        all_names.append("Noiseless")

        all_results.append(self.expected_result)
        all_errs.append(0)
        all_names.append("Exact result")

        plt.title("Ground state energy using different mitigation techniques")
        plt.xlabel("Mitigation")
        plt.ylabel("Energy")
        plt.errorbar(all_names, all_results,
                     yerr=all_errs)
        plt.show()

        return all_results, all_errs, all_names


    def compute_expectation(self, circ_params, sim=False):

        """
        Computes expectation value based on measurement results

        Args:
            counts: dict
                    key as bitstring, val as count

            G: networkx graph

        Returns:
            avg: float
                 expectation value
        """

        counts = self.get_counts(circ_params, sim=sim)
        if type(counts) == list:
            countss = counts
            res_list = []
            for counts in countss:
                avg = 0
                sum_count = 0
                for bitstring, count in counts.items():
                    obj = self.maxcut_obj(bitstring, self.G)
                    avg += obj * count
                    sum_count += count

                res_list.append(avg / sum_count)

            return res_list, countss

        else:
            avg = 0
            sum_count = 0
            for bitstring, count in counts.items():
                obj = self.maxcut_obj(bitstring, self.G)
                avg += obj * count
                sum_count += count

            result = avg / sum_count

            return result






    def mitigate_random_circuit(self, rand_params, num_shots=1000,  ZNE=None, CDR=None, vnCDR=None ,  _MCMC=False):

        self.num_shots = num_shots
        circuit = self.bind(rand_params)


        noisy_energy = self.executor(circuit)


        all_results = [noisy_energy]
        all_errs = [0]
        all_names = ["Noisy", ]  # ["Non-mitigated", "ZNE", "CDR", "vnCDR","Simulated", "Exact solution"]




        #Always do comparisons of different techniques available
        if ZNE != None:
            #doesnt happen if ZNE_during_miti happens
            print("Starting ZNE")
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
            scale_factors = ZNE["scale_factors"]
            scale_noise = ZNE["scale_noise"]
            ZNE_miti, ZNE_err = mock_execute_with_zne(circuit,
                                                      scale_factors,
                                                      self.executor,
                                                      scale_noise,
                                                      base_circuit = self.base_circuit)
            all_results.append(ZNE_miti)
            all_errs.append(ZNE_err)
            all_names.append("ZNE")

        if CDR != None:
            print("Starting CDR")
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
            num_training_circuits = CDR["num_training_circuits"]
            fraction_non_clifford = CDR["fraction_non_clifford"]
            CDR_miti, CDR_err = execute_with_cdr_alt(
                                    self.base_circuit,
                                    self.executor,
                                    simulator=self.simulator,
                                    seed=0,
                                    num_training_circuits=num_training_circuits,
                                    fraction_non_clifford=fraction_non_clifford,
                                    params=rand_params,  # if we want MCMC
                                    _MCMC=True,
                                    rand_circ=True,
                                    X_0=noisy_energy)

            all_results.append(float(CDR_miti))
            all_errs.append(CDR_err)
            all_names.append("CDR")


        if vnCDR != None:
            print("Starting vnCDR")
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
            num_training_circuits = vnCDR["num_training_circuits"]
            fraction_non_clifford = vnCDR["fraction_non_clifford"]
            scale_factors = vnCDR["scale_factors"]
            scale_noise = vnCDR["scale_noise"]
            vnCDR_miti, vnCDR_err = execute_with_cdr_alt(
                                self.base_circuit,
                                self.executor,
                                simulator=self.simulator,
                                seed=0,
                                scale_factors=scale_factors,
                                num_training_circuits=num_training_circuits,
                                scale_noise=scale_noise,
                                fraction_non_clifford=fraction_non_clifford,
                                params=rand_params,  # if we want MCMC
                                _MCMC=True,
                                rand_circ=True,
                                X_0=noisy_energy)





            all_results.append(float(vnCDR_miti))
            all_errs.append(vnCDR_err)
            all_names.append("vnCDR")

        real_energy = self.simulator(circuit)
        all_results.append(real_energy)
        all_errs.append(0)
        all_names.append("Noiseless")


        plt.title("Ground state energy using different mitigation techniques")
        plt.xlabel("Mitigation")
        plt.ylabel("Energy")

        print(all_names)
        print(all_results)
        print(all_errs)
        plt.errorbar(all_names, all_results,
                     yerr=all_errs)
        plt.show()

        return all_results, all_errs, all_names

    def bind(self, circ_params):
        # bind params to actual circuit

        circ = self.base_circuit
        theta = circ.parameters

        params = circ_params

        # print("hii")
        # print(params)

        # print(circ_params)

        param_dict = {}
        for i in range(len(theta)):
            param_dict[theta[i]] = params[i]
        # print(param_dict)
        circuit = circ.bind_parameters(param_dict)

        circuit.measure_all()
        # print(circ_params)

        return circuit

    def remove_rzz(self, circuit):
        dag = circuit_to_dag(circuit)
        dag.remove_all_ops_named("rzz")
        circuit_mod = dag_to_circuit(dag)
        return circuit_mod

    def get_counts(self, circ_params, sim=False):
        # print("getting a count")
        # can also take a circ instead of circ params
        # always remove any rzz gates, then potentially add back if add_noise is true
        # this is important for mitiq sadly
        # print("hi there")
        if type(circ_params) == type(self.base_circuit):
            #print("its a circ")
            #print(circ_params)
            circuit = circ_params


            circuit = self.remove_rzz(circuit)
            if sim == False:
                circuit.remove_final_measurements()
                #circuit = add_noise(circuit, self.noise_dict, custom_CM= self.noise_dict["coupling_map"])
                circuit = add_zz_noise_to_circ(circuit, self.noise_dict)
                circuit.measure_all()
        elif type(circ_params) == list:
            #print("its a list")
            circuits = []

            for c in circ_params:
                if type(c) == type(self.base_circuit):
                    c = self.remove_rzz(c)
                    if sim == False:
                        c.remove_final_measurements()
                        c = add_zz_noise_to_circ(c, self.noise_dict)
                        #c = add_noise(c, self.noise_dict, custom_CM=self.CM)
                        c.measure_all()
                    circuits.append(c)
                else:
                    c = self.bind(c)
                    c = self.remove_rzz(c)
                    if sim == False:
                        c.remove_final_measurements()
                        c = add_zz_noise_to_circ(c, self.noise_dict)
                        #c = add_noise(c, self.noise_dict, custom_CM=self.CM)
                        c.measure_all()
                    circuits.append(c)

            circuit = circuits



        else:
            #print("its an array")
            #print(circ_params)
            circuit = self.bind(circ_params)
            # used to be if not sim... unsure why
            circuit = self.remove_rzz(circuit)
            if sim == False:
                circuit.remove_final_measurements()

                #circuit = add_noise(circuit, self.noise_dict, custom_CM=self.noise_dict["coupling_map"])
                circuit = add_zz_noise_to_circ(circuit, self.noise_dict)
                circuit.measure_all()



        if sim == False:

            # print(sim)
            # print(circuit)

            job = qiskit.execute(
                experiments=circuit,
                backend=self.Q_inst.backend,
                noise_model=self.noise_dict["noise_model"],
                basis_gates=self.noise_dict["basis_gates"],
                optimization_level=0,  # Important to preserve folded gates.
                shots=self.num_shots)


        elif sim == True:

            job = qiskit.execute(
                experiments=circuit,
                backend=self.sim_inst.backend,
                # noise_model=noise_model,
                # basis_gates=noise_model.basis_gates,
                optimization_level=3,
                shots=self.num_shots)

        # probably add in an if statement here just because counts might
        # not always contain multiple circuits which would fuck it up

        counts = job.result().get_counts()

        return counts

    def maxcut_obj(self, x, G):
        """
        Given a bitstring as a solution, this function returns
        the number of edges shared between the two partitions
        of the graph.

        Args:
            x: str
               solution bitstring

            G: networkx graph

        Returns:
            obj: float
                 Objective
        """
        obj = 0
        for i, j in G.edges():
            if x[i] != x[j]:
                obj -= 1

        return obj

    def find_rand_circ_diff(self, rand_params, num_shots, full_output=False, scale_factors=(1,)):
        self.num_shots = num_shots
        simulator = partial(self.compute_expectation, sim=True)
        executor = partial(self.compute_expectation, sim=False)#, noise=True)

        circuit = self.bind(rand_params)  # , circuit = self.base_circuit)

        real_energy = simulator(circuit)
        noisy_energy = executor(circuit)

        if scale_factors != (1,):
            ZNE_miti, ZNE_err = mock_execute_with_zne(circuit,
                                                      scale_factors,
                                                      executor,
                                                      self.scale_noise)
            return real_energy, noisy_energy, ZNE_miti
        elif full_output:
            return real_energy, noisy_energy


        else:
            diff = abs(real_energy - noisy_energy)
            return diff
