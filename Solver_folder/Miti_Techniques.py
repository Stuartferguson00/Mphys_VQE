#Heavily adapted from mitiq, with licensing included below:

# Copyright (C) 2021 Unitary Fund
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import qiskit
import numpy as np
from scipy.optimize import curve_fit
from mitiq.cdr import (
    generate_training_circuits,
    #linear_fit_function,
    linear_fit_function_no_intercept,
)
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.inference import Factory, RichardsonFactory,LinearFactory, ExpFactory
from .MCMC_ import MCMC
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit







# a suite of fitting functions used at one time or another for testing

def linear_fit_function(x_data, params):
    fit = sum(a * x for a, x in zip(params, x_data)) + params[-1]
    return fit

def linear_fit_fnc_err(x_data, params_err):
    err = np.sqrt(sum(a ** 2 * x ** 2 for a, x in zip(params_err, x_data)))
    return err

def linear_fit_function_no_intercept(x_data, params):
    return sum(a * x for a, x in zip(params, x_data))

def bb_linear_fit(x, params):
    m, c = params
    return m * x + c

def bb_linear_fit_no_c(x, params):
    m = params
    return m * x

def bb_linear_fit_no_c_err(m_err, x):
    return np.sqrt(m_err ** 2 * x ** 2)

def bb_linear_fit_no_c_err(m_err, x):
    return np.sqrt(m_err ** 2 * x ** 2)

def bb_linear_fit_err(param_err, x):
    m_err, c_err = param_err
    return np.sqrt(m_err ** 2 * x ** 2 + c_err ** 2 * x ** 2)







def execute_with_cdr_alt(circuit,
    executor,
    simulator,
    params = None,
    scale_factors = (1,),
    num_training_circuits = 10,
    scale_noise = fold_gates_at_random, 
    fraction_non_clifford = 0.1,
    fit_function = linear_fit_function,
    _MCMC = False,
    rand_circ = None,
    X_0 = None ):
    
    

    
    dag = circuit_to_dag(circuit)

    dag.remove_all_ops_named("rzz")
    circuit = dag_to_circuit(dag)
    
    

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
        )
    else:
        #generate training circuits using MCMC
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


    # compute noisy expectation values
    results, counts = executor(to_run)
    noisy_results = np.array(results).reshape(all_circuits_shape)

    inp = noisy_results
    consts = np.ones((noisy_results.shape[0]))
    consts = consts[..., np.newaxis]
    inp = np.hstack((inp, consts))

    #used ols, but had same output as scipy
    ols = sm.regression.linear_model.OLS(ideal_results, inp[1:, :])
    ols_result = ols.fit()
    miti_err_ols = linear_fit_fnc_err(inp[0, :].T, ols_result.bse)
    miti_err = miti_err_ols
    fitted_params = ols_result.params
    mitigated_energy = ols_result.predict(inp[0, :].T)  # fit_function(noisy_results[0, :], fitted_params)


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



























