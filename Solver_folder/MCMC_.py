import copy
from sklearn.linear_model import LinearRegression
from typing import Any, Callable, Optional, Sequence, Union, List
from functools import wraps
import numpy as np
from scipy.optimize import curve_fit

#import statsmodels.regression.linear_model as sm
from mitiq import Executor, Observable, QPROGRAM, QuantumResult
from mitiq.cdr import (
    generate_training_circuits,
    #linear_fit_function,
    linear_fit_function_no_intercept,
)
from mitiq.cdr.clifford_utils import is_clifford
from mitiq.zne.scaling import fold_gates_at_random
from matplotlib import pyplot as plt
from mitiq.zne.inference import Factory, RichardsonFactory,LinearFactory

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit import ParameterVector, QuantumCircuit



class MCMC:
      
    """
    A class to conduct Markov Chain Monte Carlo .


    Parameters
    ----------
    N : int
        Number of non-clifford gates allowed in sample circuits
    
    base_circuit : Qiskit quantum circuit object
        Qiskit Paramatrised Quantum Circuit with parameter values unassigned


    simulator : func
        function that returns the noiseless simulated expectation value of a circuit

    executor : func
        function that returns the noisey (could be simulated or real QC) expectation value of a circuit

    X_s : float
        value representing the deviation of sample likelihood required. (papers report this between 0.05 and 0.2)

    Returns
    -------

    """
    
    
    
    
    
    
    
    
    
    
    
    
            
    def __init__(self,fraction_non_clifford,base_circuit,simulator, executor, X_s = 0.18):
        
        '''
        Initialisation function of the class
        
        Parameters
        ----------
        N : int
            Number of non-clifford gates allowed in sample circuits
            
        base_circuit : Qiskit quantum circuit object
            Qiskit Paramatrised Quantum Circuit with parameter values unassigned

        simulator : func
            function that returns the noiseless simulated expectation value of a circuit

        executor : func
            function that returns the noisy (could be simulated or real QC) expectation value of a circuit
            
        X_s : float
            value representing the deviation of sample likelihood required. (papers report this between 0.05 and 0.2)
            

        Returns
        
        -------
        
        
        '''

        
        self.fraction_non_clifford = fraction_non_clifford
        self.base_circuit = base_circuit
        self.simulator = simulator
        self.executor = executor
        self.X_s = X_s
        
        


        
    """   
    def bind(self, circ_params):
        #bind params to actual circuit
        circ = copy.deepcopy(self.base_circuit)
        theta = circ.parameters
        params = circ_params

        param_dict = {}
        for i in range(len(theta)):    
            param_dict[theta[i]] = params[i]

        circuit = circ.bind_parameters(param_dict)

        circuit.measure_all()

        return circuit
    """

    
    
    
    def bind(self,circ_params, other_circuit = None):
        #QAOA bind = true replaces the circuit with the same circuit but with the maximum number of paramaters. 
        #It splits paramaters that paramatrize more than one gate into many paramaters controlling one gate each. Necessary for QAOA.
        
        
        #bind params to actual circuit
        if other_circuit == None:
            circ = copy.deepcopy(self.base_circuit)
            
        else:
            circ = copy.deepcopy(other_circuit)
            
            
        
        
        theta = circ.parameters
        params = circ_params

        param_dict = {}
        for i in range(len(theta)): 

            param_dict[theta[i]] = params[i]

        

        if other_circuit  == None:
            circuit = circ.bind_parameters(param_dict)
        else:
            circuit = circ.assign_parameters(param_dict)
            
       
            
        circuit.measure_all()

        return circuit

    
    
    
    
    
    
    def make_QAOA_many_param(self, circuit):
        #redesign circuit, replacing all parametrized gates with a individual paramaters
        n = circuit.num_qubits


        
        qc = QuantumCircuit(n)

        gates = []
        qargs = []
        num_qubits = []
        count = 0
        param_list = []
        for gate, qarg, _ in circuit._data:

            num_qubits.append(gate.num_qubits)
            gates.append(gate.name)
            qargs.append(qarg)
            if gate.name == "rz" or gate.name == "rx" or gate.name == "ry":
                
                param_list.append(float(gate.params[0]))
                count+=1

        theta_2 =  ParameterVector('a', count)


        count_2 = 0
        for i,  gate in enumerate(gates):
            if gate == "h":
                
                qc.h(qargs[i][0])
            elif gate == "rz":

                qc.rz(theta_2[count_2], qargs[i][0])
                count_2+=1

            elif gate == "rx":
                qc.rx(theta_2[count_2], qargs[i][0])
                count_2+=1
                
            elif gate == "ry":
                qc.ry(theta_2[count_2], qargs[i][0])
                count_2+=1


            elif gate == "cx":
                qc.cx( qargs[i][0], qargs[i][1])
            elif gate == "measure":
                pass
                #qc.measure( qargs[i][0],qargs[i][0])
            else:
                pass



        #qc = bind(param_list, QAOA_bind = True, other_circuit = qc)


        return qc, param_list



    
    
    

    def likelihood(self, X):
        
        """
        Returns the likelhood of a particular value X wrt to X_s
        
        Parameters
        ----------
        X : Expectation value of circuit

        Returns
        -------
        L : float
            likelihood of X
        """

        
        if self.rand_circ:
            #unminimised cirucit
            L = np.exp((-(X-self.X_0)**(2))/self.X_s)
            return L
        
        else:
            #minimised circuit
            #r = np.exp(-(X+2)**2/(X_s)**2)
            L = np.exp(-(X)/(self.X_s))
            return L
    
    
    def acceptance_ratio(self,p, p_new):
        """
        Returns the ratio of acceptance for a given circuit against the previous accepted circuit
        
        Parameters
        ----------
        p : float
            The expectation value of the last circuit to be accepted
            
        p_new : float
            The expectation value of the circuit which is being evaluated currently

        Returns
        -------
        
        A : float
            Acceptance ratio of new expectation value
        
        """
        
        
        A = min(1, (((self.likelihood(p_new) / self.likelihood(p)) * ((p_new) / (p))  )))
        return A
        
    

    def prior(self,circ):
        """
        Find the prior (expectation value of a circuit)
        
        Parameters
        ----------
        circ : Qiskit Quantum Circuit object
            Qiskit Quantum Circuit with parameter values assigned
            
        Returns
        -------
        
        X : float
            Simualted expectation value of the circuit
        
        """

        
        X = self.simulator(circ)
        return X
    
    
    
    
    
    
    
    
    
    def find_training_circuits_MCMC(self, circ_params, num_training_circuits, X_0 = None, rand_circ = False):
        """
        Function to find similar, near-clifford training circuits for a given circuit.
        
        Parameters
        ----------
    
        circ_params : list/array of floats
            list of angles corresponding to each rotation gate in the parametrised base circuit
        num_training_circuits : int
            number of training circuits the function should produce
            
        Returns
        -------
        
        training_circuits : list of qiskit quantum circuit objects
            list where each instance is a qiskit quantum circuit corresponding to a training circuit
                
        """
        self.rand_circ = rand_circ
        
        if rand_circ:
            if X_0 !=None:
                self.X_0 = X_0
            else:
                print(" for a random circuit, you must provide a centroid")

        training_circuits = []
        i = 0

        # Generate training circuit parameters
        #update them according to CDR paper
        
        #copy original circuit parameters
        
        
        
        circuit = self.base_circuit
        circuit = self.bind(circ_params)
        
        
        circuit, circ_params = self.make_QAOA_many_param(circuit)
        
        circ_params = np.array(circ_params)
        
        old_params = copy.deepcopy(circ_params)

        
        
        
        num_class_sims = 0
        n = circuit.num_parameters

        
        self.N = round(n*self.fraction_non_clifford)
        not_used_count = 0
        #itterate untill we have enough training circuits
        while i<num_training_circuits:
            #If first loop, use N
            if i == 0:
                new_params = self.replace_with_clifford(old_params)
            else:
                new_params = self.update_params(old_params, circ_params)
            
            
            
            #attach new params to a training circuit
            circ = self.bind(new_params, other_circuit = circuit )
            
            
            
            
            
            #find prior
            X = self.prior(circ)
            
            num_class_sims +=1
            
            
            #if it is first loop set as the first circuit
            #no burn in atm
            if i == 0:
                X_old = X
                old_prams = new_params
                training_circuits.append(circ)
                i+=1
                
                #plt.plot(X,self.likelihood(np.array(X)), "rx")
                #plt.yscale('symlog')
                
            else:
                #maybe add in a tally so we know how many are rejected ie how long it takes
                a_ratio = self.acceptance_ratio(X_old,X)
                u = np.random.random_sample()
                if u < a_ratio:
                    old_prams = new_params
                    X_old = X
                    training_circuits.append(circ)
                    i +=1

                    #plt.plot(X,self.likelihood(np.array(X)), "rx")

                    #if i %10==0:
                    #    print(str(i)+ " training circuits")
                        
                else:
                    not_used_count +=1
                    #if not_used_count %100==0:
                    #    print(str(not_used_count)+ " not used circuits")
                        

        x = np.arange(-2.5,1,0.01)


        #for x_ in x:
        #    #plt.plot(x_, self.likelihood(np.array([x_])), "bx", alpha = 0.1)  
            
        #plt.title("")
        #plt.xlabel("expectation value")
        #plt.ylabel("likelihood of being accepted")

        #plt.show()
        return training_circuits

    
           
        
        
    def replace_with_clifford(self,params):
        
        

        
        #find n-N random gates to swap to clifford
        inds = np.sort(np.random.choice(len(params), size=len(params)-self.N, 
                                replace = False))

        #assign random clifford rotation to the relevant gates
        clifford = np.random.randint(0,4, len(params))*(np.pi/2)
        

        
        params[inds] = clifford[inds] 

        #define class variables for use elsewhere
        self.cliff_inds = inds  
        self.non_cliff_inds = np.setdiff1d(np.arange(0,len(params)),self.cliff_inds)
        new_params = params
        return new_params
        
        
        
    
    def update_params(self, old_params, orig_params, n_p = 2):
        """
        Finds a new training circuit to test, updates according to CDR paper
        
        Parameters
        ----------
        old_params : list/array of floats
            list of angles corresponding to each rotation gate in the parametrised base circuit,
            of the last training circuit accepted

        orig_params : list/array of floats
            list of angles corresponding to each rotation gate in the parametrised base circuit,
            of the original circuit supplied
            
        n_p : int
            number of gates to change in this update


        Returns
        -------
        
        new_params : new_params : list/array of floats
            list of angles corresponding to each rotation gate in the parametrised base circuit, 
            where some gates have now been replaced with clifford gates and clifford gates 
            replaced with the original supplied rotations
        
        
        """

        
        
        
        #change n_p cliffords back to original parameters
        change_to_non_cliff_inds = np.random.choice(len(self.cliff_inds), 
                                                size=n_p, replace = False)
        #change n_p non-cliffords to clifford
        change_to_cliff_inds = np.random.choice(len(self.non_cliff_inds), 
                                                size=n_p, replace = False) 
        
        #assign new clifford gates to selected gates
        new_params = copy.deepcopy(old_params)
        for i in change_to_cliff_inds:
            new_params[i] = np.random.randint(0,4, 1)*(np.pi/2)
        
        #reassign original non-clifford gate selected gates
        for i in change_to_non_cliff_inds:
            new_params[i] = orig_params[i]
        
        
        return new_params
     
    











