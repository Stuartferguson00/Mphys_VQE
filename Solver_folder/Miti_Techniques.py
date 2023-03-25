# Copyright (C) 2021 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""API for using Clifford Data Regression (CDR) error mitigation."""
import copy
import qiskit

from qiskit.opflow import DictStateFn
from sklearn.linear_model import LinearRegression
from typing import Any, Callable, Optional, Sequence, Union, List
from functools import wraps
import numpy as np
from scipy.optimize import curve_fit

#import statsmodels.regression.linear_model as sm
import statsmodels.api as sm
from mitiq import Executor, Observable, QPROGRAM, QuantumResult
from mitiq.cdr import (
    generate_training_circuits,
    #linear_fit_function,
    linear_fit_function_no_intercept,
)
from mitiq.cdr.clifford_utils import is_clifford
from mitiq.zne.scaling import fold_gates_at_random
from matplotlib import pyplot as plt
from mitiq.zne.inference import Factory, RichardsonFactory,LinearFactory, ExpFactory
from .MCMC_ import MCMC

from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.tools.visualization import dag_drawer
from qiskit.circuit.library import CHGate, U2Gate, CXGate, RZZGate
from qiskit.converters import dag_to_circuit





def linear_fit_function(x_data, params):
    fit = sum(a * x for a, x in zip(params, x_data)) + params[-1]
    return fit

def linear_fit_fnc_err(x_data, params_err):
    err = np.sqrt(sum(a**2 * x**2 for a, x in zip(params_err, x_data))) 
    return err

def linear_fit_function_no_intercept(x_data, params):    
    return sum(a * x for a, x in zip(params, x_data))


def bb_linear_fit(x, params):
    m,c = params
    return m*x+c
    
def bb_linear_fit_no_c(x, params):
    m = params
    return m*x

def bb_linear_fit_no_c_err(m_err, x):
    return np.sqrt(m_err**2*x**2)

def bb_linear_fit_no_c_err(m_err, x):
    return np.sqrt(m_err**2*x**2)

def bb_linear_fit_err(param_err, x):
    m_err, c_err = param_err
    return np.sqrt(m_err**2*x**2+c_err**2*x**2)



def execute_with_cdr_alt(circuit,
    executor,
    simulator,
    params = None,
    seed = 1000,
    scale_factors = (1,),
    num_training_circuits = 10,
    scale_noise = fold_gates_at_random, 
    fraction_non_clifford = 0.1,
    fit_function = linear_fit_function,
    observable = None,
    _MCMC = False,
    rand_circ = None,
    X_0 = None ):
    
    
    
    """
        
        
        Parameters
        ----------
        
        Returns
        -------
        
        
    """
    
    
    
    dag = circuit_to_dag(circuit)

    dag.remove_all_ops_named("rzz")
    #dag.remove_all_ops_named("ry")
    #dag.remove_all_ops_named("barrier")
    circuit = dag_to_circuit(dag)
    
    
    
    #making sure the correct fit function is being used
    if scale_factors == (1,):

        #fit_function = linear_fit_function  #weirtd error here, not sure why
        fit_function = bb_linear_fit
        num_fit_parameters = 2
    else:
        fit_function = bb_linear_fit
        num_fit_parameters = 3
        #fit_function = linear_fit_function_no_intercept
        

    if _MCMC == False:
        # Generate training circuits normally
        
        theta = circuit.parameters
        param_dict = {}
        for i in range(len(theta)):    
            param_dict[theta[i]] = params[i]
        circuit = circuit.bind_parameters(param_dict)

        circuit.measure_all()
        training_circuits = generate_training_circuits(
            circuit,
            num_training_circuits,
            fraction_non_clifford,
            #method_select,
            #method_replace,
            #random_state,
            #kwargs=kwargs_for_training_set_generation,
        )
    else:
        #generate training circuits using MCMC
        #n = circuit.num_parameters
        #N = round(n*fraction_non_clifford)
        mcmc = MCMC(fraction_non_clifford,circuit, simulator, executor)

        training_circuits = mcmc.find_training_circuits_MCMC( params, num_training_circuits, rand_circ = rand_circ, X_0 =X_0)
        
    
        #bind params to actual circuit
        theta = circuit.parameters
        param_dict = {}
        for i in range(len(theta)):    
            param_dict[theta[i]] = params[i]
        circuit = circuit.bind_parameters(param_dict)

        circuit.measure_all()
        



    
    
    # Optionally scale noise in circuits ie for znCDR
    all_circuits = [
        [scale_noise(c, s) for s in scale_factors]
        for c in [circuit] + training_circuits  
    ]

    to_run = [circuit for circuits in all_circuits for circuit in circuits]
    all_circuits_shape = (len(all_circuits), len(all_circuits[0]))


    #compute simulated expectation values
    results, counts = simulator([circuit,]+training_circuits)
    ideal_results = results[1:]
    
    
    """
    print("results")
    print(results)
    print("ideal_results")
    print(ideal_results)
    """
    
    """
    y = []

    for c in counts:
        c_mat = DictStateFn(c).to_matrix()
        y.append(c_mat.real)
        
    y_e_sim = results
    """
    
    
    
    
    
    
    #just bypassing this for now!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if scale_factors == None:#(1,):
        
        #compute noisy expectation values
        results, counts = executor(to_run)

        noisy_results = np.array(results).reshape(all_circuits_shape)
        
        
        """
        y_e_noisy = noisy_results
        
        X = []
        for c in counts:
            c_mat = DictStateFn(c).to_matrix()
            X.append(c_mat.real)
        """
        

        #if no scale factors included, then it is just CDR so linear fit:
        fitted_params, pcov, *_ = curve_fit(
            lambda x, *params: fit_function(x, params),
            noisy_results[1:, :].T[0],
            ideal_results,
            p0=np.zeros(num_fit_parameters),
            full_output=True, 
            xtol = 1.49012e-13)
        
        perr = np.sqrt(np.diag(pcov))
        miti_err = linear_fit_fnc_err(noisy_results[0, :].T, perr)
        mitigated_energy = fit_function(noisy_results[0, :], fitted_params)
        
        
        """

        ideal_circ_result = simulator(circuit)
        noisy_circ_result = executor(circuit)
        x = np.array([np.arange(ideal_circ_result - 0.1,np.max(noisy_results),0.1).T])
        y = fit_function(x, fitted_params)
        

        
        
        all_y_err = linear_fit_fnc_err(x, perr)
        
        plt.fill_between(x[0], y[0] - all_y_err[0], y[0] + all_y_err[0], color='b', alpha=0.2)


        plt.plot(x[0],y[0], "r", label = "linear_fit")
        
        
        
        print("hi")
        print(ideal_results)
        
        
        #print(noisy_results[1:, :][0], ideal_results)
        
        plt.plot(noisy_results[1:, :].T[0], ideal_results,"bx", label = "original_results")
        #plt.plot(noisy_results[1:, :].T[0], ideal_results,"bx", label = "original_results")





        plt.errorbar(noisy_results[0, :],fit_function(
                        noisy_results[0, :],fitted_params),
                        yerr = miti_err,
                        marker="x", color = "b",
                        label = "actual_circ_mitigated")
        
        
        plt.plot(noisy_results[0, :], ideal_circ_result,
                         "kx", label = "actual_circ_ideal")
        plt.plot(noisy_results[0, :], noisy_results[0, :], 
                         "gx", label = "actual_circ_non_mitigated")
        plt.xlabel("noisy results")
        plt.ylabel("ideal results")
        plt.legend()
        plt.title("CDR Error mitigation ")
        plt.show()
        """
        
        
        
        """
        np.save('X.npy', X) 
        np.save('y.npy', y) 
        np.save('y_e_noisy.npy', y_e_noisy) 
        np.save('y_e_sim.npy', y_e_sim) 
        """
        
        #print(miti_err)
        
        return mitigated_energy, miti_err


    
    
    else:
        
        
        """
        fitted_params, pcov, *_ = curve_fit(
            lambda x, *params: fit_function(x, params),
            noisy_results[1:, :].T,
            ideal_results,
            p0=np.zeros(num_fit_parameters),
            full_output=True, 
           )
        perr = np.sqrt(np.diag(pcov))
        mitigated_energy = fit_function(noisy_results[0, :], fitted_params)
        
        """
        # compute noisy expectation values
        results, counts = executor(to_run)
        noisy_results = np.array(results).reshape(all_circuits_shape)

        inp = noisy_results
        #print(inp)
        consts = np.ones((noisy_results.shape[0]))
        consts = consts[..., np.newaxis]
        #print(consts)
        inp = np.hstack((inp,consts))
        #print(inp)
        ols = sm.regression.linear_model.OLS(ideal_results, inp[1:, :])
        ols_result = ols.fit()
        # and covariance estimates
        #ols_result.cov_HC0
        #perr_ols = np.sqrt(np.diag(ols_result.cov_HC0))
        miti_err_ols = linear_fit_fnc_err(inp[0, :].T, ols_result.bse)
        miti_err = miti_err_ols
        fitted_params = ols_result.params
        #print(noisy_results[1:, :])
        #print(noisy_results[0, :].T)
        mitigated_energy = ols_result.predict(inp[0, :].T)#fit_function(noisy_results[0, :], fitted_params)

        #print(ideal_results)
        #print(noisy_results[1:, :])
        #print(fitted_params)
        #print(noisy_results[0, :].T)
        
        
        
        
        """
        #perform ZNE on each training circuit        
        ZNE_results = []
        ZNE_err = []
        for circ in training_circuits:
            
            res, err = mock_execute_with_zne(circ,  scale_factors, executor, scale_noise)
            ZNE_results.append(res)
            ZNE_err.append(err)
        
        
        plt.title("ZNE vs ideal")
        plt.ylabel("ideal results")
        plt.xlabel("ZNE mitigated results")
        plt.plot(ZNE_results, ideal_results, "bx")
        
        
        ZNE_results = np.array(ZNE_results)
        ideal_results= np.array(ideal_results)
        
        
        
        num_fit_parameters = 2        
        #inform the ZNE od the ideal results from CDR
        fitted_params, pcov, *_ = curve_fit(
            lambda x, *params: fit_function(x, params),
            ZNE_results,
            ideal_results,
            p0=np.ones(num_fit_parameters),
            full_output=True, 
            xtol = 1.49012e-13)
        
          

        
        
        
        #ZNE on actual circuit
        ZNE_circ_result, ZNE_circ_err =  mock_execute_with_zne(circuit,  scale_factors, executor, scale_noise)
        
        #find errors and mitigated value from CDR fit
        perr = np.sqrt(np.diag(pcov))
        miti_err = linear_fit_fnc_err([ZNE_circ_result], perr)
        mitigated_energy = fit_function(ZNE_circ_result, fitted_params)
        """

        """
        
        ideal_circ_result = simulator(circuit)
        noisy_circ_result = executor(circuit)
        x = np.array([np.arange(ideal_circ_result - 0.1,np.max(ZNE_results),0.1).T])
        y = fit_function(x, fitted_params)
        all_y_err = linear_fit_fnc_err(x, perr)
        
        
        plt.fill_between(x[0], y[0]- all_y_err, y[0] + all_y_err, color='b', alpha=0.2)

        
        print("helloooooooo")
        plt.plot(x[0],y[0], "r", label = "linear_fit")
        plt.plot(ZNE_results, ideal_results,"bx", label = "original_results")

        



        plt.errorbar(ZNE_circ_result,fit_function(
                        ZNE_circ_result,fitted_params),
                        yerr = miti_err,
                        marker="x", color = "k",
                        label = "actual_circ_mitigated")
        
        
        plt.plot(ZNE_circ_result, ideal_circ_result,
                         "kx", label = "actual_circ_ideal")
        plt.plot(ZNE_circ_result, ZNE_circ_result, 
                         "gx", label = "actual_circ_non_mitigated")
        plt.xlabel("noisy results")
        plt.ylabel("ideal results")
        #plt.legend()
        plt.title("CDR Error mitigation ")
        plt.show()
        
        
        print(miti_err)
        """
        return mitigated_energy, miti_err
        
        
        






        
def mock_execute_with_zne(circuit,  scale_factors, executor, scale_noise = fold_gates_at_random, base_circuit = None,return_err = True):
    
    
    """


    Parameters
    ----------

    Returns
    -------



    """
    #print("in mock")
    if type(circuit) != qiskit.circuit.quantumcircuit.QuantumCircuit:
        
        #bind params to actual circuit
        theta = base_circuit.parameters
        param_dict = {}
        params = circuit
        for i in range(len(theta)):    
            param_dict[theta[i]] = params[i]
        circuit = base_circuit.bind_parameters(param_dict)

        

        
    dag = circuit_to_dag(circuit)

    dag.remove_all_ops_named("rzz")
    #dag.remove_all_ops_named("measure")
    
    #dag.remove_all_ops_named("ry")
    #dag.remove_all_ops_named("barrier")
    circuit = dag_to_circuit(dag)
    circuit.remove_final_measurements()
    
    
    scale_factors = tuple(scale_factors)

    factory = LinearFactory(scale_factors)

    #factory = ExpFactory(scale_factors)
    
    
    
    
    circuit.measure_all()

    factory.run(circuit, executor, scale_noise =scale_noise)
    #print(factory.res)
    #miti_res = factory.extrapolate(scale_factors,all_res)
    miti_res= factory.reduce()



    f = factory.plot_fit()
    f.show()






    if return_err:
        err = factory.get_zero_noise_limit_error()

        return miti_res, err
    else:
        return miti_res
        
        
        




    
    
    
    
    
    
    
    
   
        
        
    
    
   



    
    
    
    
    
    
    
    
  
    
