import qiskit_aer
import matplotlib.pyplot as plt
from qiskit.compiler import transpile
import numpy as np
from qiskit.providers.aer.noise import NoiseModel
from qiskit.transpiler import CouplingMap 
from collections import OrderedDict
from qiskit.circuit import ParameterVector, QuantumCircuit, ParameterExpression
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.tools.visualization import dag_drawer
from qiskit.circuit.library import CHGate, U2Gate, CXGate, RZZGate
from qiskit.converters import dag_to_circuit
plt.style.use('default')









def hardware_efficient_build(no_qubits, depth):

    """
    Function to build a hardware efficient ansatz

    Parameters
    ----------
    no_qubits : int
        number of qubits

    depth :
        number of layers of Ry gates

    Returns
    -------

    circuit : obj
        Qiskit.QuantumCircuit object of the hardware efficient ansatz
    """


    theta = ParameterVector('θ', no_qubits*depth)
    qreg_q = QuantumRegister(no_qubits, 'q')
    circuit = QuantumCircuit(qreg_q)

    
    count = 0



    for i in range(no_qubits*depth):
        if count == no_qubits:
            count = 0
            for j in range(0,no_qubits-1):
                circuit.cx(qreg_q[j], qreg_q[j+1])
            
        
        circuit.ry(theta[i], qreg_q[count])
    

        count+=1
        
        
        
    return circuit








def remove_idle_qwires(circ):

    """
    Helper function to remove idle wires from a quantum circuit

    Parameters
    ----------
    circ : obj
        Qiskit.QuantumCircuit onject

    Returns
    -------

    circ_no_idle : obj
        Qiskit.QuantumCircuit object, with no idle wires

    """

    dag = circuit_to_dag(circ)

    idle_wires = list(dag.idle_wires())
    for w in idle_wires:
        dag._remove_idle_wire(w)
        dag.qubits.remove(w)

    dag.qregs = OrderedDict()

    circ_no_idle =  dag_to_circuit(dag)
    return circ_no_idle





    


def apply_all_rzz(c_lst, old_dag,q, theta, n):


    """

    Helper function to apply Rzz gates between all coupled qubits


    Parameters
    ----------
    c_lst : list
        list of qubit couplings of form: [[0,1],[0,2]]

    old_dag : obj
        Qiskit.Dag object representing a quantum circuit
    q : obj
        Qiskit.QuantumRegister object

    theta : obj
        Qiskit.ParameterVector object

    n : obj
        numbe rof qubits in circuit

    Returns
    -------

    """

    for count, i in enumerate(c_lst):
        if count/2 == count/2:
            if i[0] < n and i[1] < n:
                pass
                old_dag.apply_operation_back(RZZGate(theta), qargs=[q[i[0]],q[i[1]]])
                #old_dag.apply_operation_back(RZZGate(theta), qargs=[q[i[1]],q[i[0]]])
    new_dag = old_dag
    return new_dag


def add_zz_noise_to_circ(circuit, noise_dict):

    """

    Helper function to apply ZZ noise model, both idle and driven according to noise_dict


    Parameters
    ----------
    circuit : obj
        Qiskit.QuantumCircuit object

    noise_dict : dict
        standard noise dictionary

    Returns
    -------

    circuit : obj
        Qiskit.QuantumCircuit object with ZZ noise as dictated by noise_dict

    """

    if noise_dict["idle_zz"] != 0:
        circuit = apply_zz_idle_to_circuit(circuit,
                                           noise_dict["coupling_map"],
                                           noise_dict["basis_gates"], noise_dict["idle_zz"])

    if noise_dict["driven_zz"] != 0:

        circuit = apply_zz_driven_to_circuit(circuit,
                                             noise_dict["coupling_map"],
                                             noise_dict["basis_gates"], noise_dict["driven_zz"], only_one = False)
    return circuit



    
def apply_zz_idle_to_circuit(circuit, c_map, custom_BG, theta):

    """

    Helper function to apply Rzz gates between all coupled qubits every layer of the circuit



    Parameters
    ----------
    circuit: obj
        Qiskit.QuantumCircuit object

    c_map : obj
        Qiskit.CouplingMap object

    custom_BG : obj
        Qiskit.BasisGates object

    theta : obj
        Qiskit.PatrameterVector object

    Returns
    -------
    circ : obj
        Qiskit.QuantumCircuit object with ZZ noise as dictated by noise_dict

    """


    n = circuit.num_qubits
    dag = circuit_to_dag(circuit)
    
    
    c_map_edges= c_map.get_edges()
    #print(c_map_edges)
    #be careful if its a real qiskit c_map object or just a list
    c_map_alt = []
    count = 0

    for i in range(0,len(c_map_edges)):
        count +=1

        if count//2 == count/2:
            c_map_alt.append(c_map_edges[i])
    #print(c_map_alt)
    layers = dag.layers()



    for count,j in enumerate(layers):

        if count == 0:
            _dag = DAGCircuit()
            

            #c = ClassicalRegister(1, "c")         
            q = QuantumRegister(n, "q")

            _dag.add_qreg(q)
            #_dag.add_creg(c)

            #print(j["partition"])
            _dag.compose(j["graph"])       
            _dag = apply_all_rzz(c_map_alt, _dag, q, theta, n)

        else:
            _dag.compose(j["graph"])
            _dag = apply_all_rzz(c_map_alt, _dag, q, theta, n)

    circ = dag_to_circuit(_dag)

    
    circ_d  = transpile(circ, 
                        coupling_map = c_map,
                        #coupling_map = coupling_m,
                        basis_gates=custom_BG, 
                        optimization_level=0)  



    return circ_d





def depolarising_NM(id_err, x_err, sx_err, cx_err, meas_err, reset_err):

    """

    Helper function to create depolarising noise model


    Parameters
    ----------
    id_err : float
        error I on gate

    x_err: float
        error x on gate

    sx_err: float
        error sx on gate

    cx_err: float
        error on cx gate

    meas_err: float
        error on measuremnt

    reset_err: float
        error on greset


    Returns
    -------

    custom_NM : obj
        Qiskit.NoiseModel object

    custom_BG : obj
        Qiskit.BasisGates object

    """

    noise_model = NoiseModel()



    # Add depolarizing error to all single qubit u1, u2, u3 gates

    ID_err = qiskit_aer.noise.depolarizing_error(id_err, 1)
    X_err = qiskit_aer.noise.depolarizing_error(x_err, 1)
    SX_err = qiskit_aer.noise.depolarizing_error(sx_err, 1)
    CX_err = qiskit_aer.noise.depolarizing_error(cx_err, 2)
    MEAS_err = qiskit_aer.noise.depolarizing_error(meas_err, 1)
    RESET_err = qiskit_aer.noise.depolarizing_error(reset_err, 1)

    noise_model.add_basis_gates(['reset'])
    noise_model.add_all_qubit_quantum_error(ID_err, ["id"])
    noise_model.add_all_qubit_quantum_error(X_err, ["x"])
    noise_model.add_all_qubit_quantum_error(SX_err, ["sx"])
    noise_model.add_all_qubit_quantum_error(CX_err, ["cx"])
    noise_model.add_all_qubit_quantum_error(MEAS_err, ["measure"])
    noise_model.add_all_qubit_quantum_error(RESET_err, ["reset"])

    noise_model.add_basis_gates(['rzz'])
    return noise_model, noise_model.basis_gates




def apply_zz_driven_to_circuit(circuit, c_map,custom_BG, theta, only_one = False):


    """

    Helper function to apply Rzz gates between all qubits driven by CNOT



    Parameters
    ----------
    circuit: obj
        Qiskit.QuantumCircuit object

    c_map : obj
        Qiskit.CouplingMap object

    custom_BG : obj
        Qiskit.BasisGates object

    theta : obj
        Qiskit.PatrameterVector object

    Returns
    -------
    circ : obj
        Qiskit.QuantumCircuit object with driven ZZ noise

    """



    n = circuit.num_qubits
    dag = circuit_to_dag(circuit)
    
    
    c_map_edges= c_map.get_edges()
    
    
    dag = circuit_to_dag(circuit)

    
    
    cx_node = dag.op_nodes(op=CXGate)#.pop()
    
    c_map_edges =np.asarray(c_map_edges)

    new_cme = []
    for i in c_map_edges:
        
        if i[0] <n and i[1] <n:
            new_cme.append(i)
    new_cme = np.asarray(new_cme)

    
    layers = dag.layers()

    new_dag = DAGCircuit()
    q = QuantumRegister(n, "q")
    new_dag.add_qreg(q)

    for count,j in enumerate(layers):

        
        mini_dag = DAGCircuit()
        p = QuantumRegister(n, "p")
        mini_dag.add_qreg(p)
        mini_dag.compose(j["graph"]) 
        mini_dag  = replace_CX_with_driven_ZZ(mini_dag, new_cme, theta, p, only_one = only_one)
        
        new_dag.compose(mini_dag)
        
        

        
        
    circ = dag_to_circuit(new_dag)
    
    
    
    circ_d  = transpile(circ, 
                        coupling_map = c_map,
                        #coupling_map = coupling_m,
                        basis_gates=custom_BG, 
                        optimization_level=0)  



    return circ
    
    
def replace_CX_with_driven_ZZ(_dag, cme, theta, q, only_one = False):

    """

    Helper function to replace a CNOT dag with a dag consisting of a cnot and any other Rzz gates required for driven ZZ nosie

    Parameters
    ----------
    _dag
    cme
    theta
    q
    only_one

    Returns
    -------

    """


    cx_node = _dag.op_nodes(op=CXGate)#.pop()
    #print("start")
    #print(cx_node)
    for node in cx_node:
        #print("wtf")
        #print(node.qargs)
        q_0 = node.qargs[0]
        q_1 = node.qargs[1]
        relevant_edges = []
        count = 0
        for edge in cme:
            #print(count)
            """
            if count == 1:
                count = 0
                #so that each edge is only done once (not twice)
                pass
            else:
                count +=1
            """

            #print(edge)


            if q_0.index in edge and q_1.index in edge:
                #print("hi")
                pass

            #elif q_0.index in edge or q_1.index in edge:

            #Ie only do RZZ for pulse leakage (when target pulse overspills into other qubits than target)
            #so no driven ZZ crosstalk from qubit that is target CNOT
            elif q_0.index == edge[0]:
                #print("hello")
                #if doing ramsey circuit and onl want one connection
                if only_one:
                    if edge[0]>1 or edge[1]>1:
                        pass
                    else:
                        _dag.apply_operation_back(RZZGate(theta), qargs=[q[edge[0]], q[edge[1]]])
                else:
                    _dag.apply_operation_back(RZZGate(theta), qargs=[q[edge[0]],q[edge[1]]])
                #_dag.apply_operation_back(RZZGate(theta), qargs=[q_0,q_1])

    return _dag




def Ramsey_build(depth, f, initial_state = "0", one_qubit = False):



    """

    Function to build a circuit for ramsey experiment

    Parameters
    ----------
    depth : int
         number of identity gates in delay

    f : float
        frequencty bias to add

    initial_state : str
        "0", "1" or "+", depending on the starting state of spectator qubit

    one_qubit : bool
        True for no spectator qubit

    Returns
    -------

    """

    #initial state can be +, 0 or 1, expected in string format
    if one_qubit:
        qc = QuantumCircuit(1)
        #cr = ClassicalRegister(1, "c")
        #qc.add_register(cr)
        #qc.rx(np.pi/2,1)
        #qc.h(0)
        qc.rx(np.pi/2,0)
        #qc.sx(1)

        qc.barrier()
        for d in range(depth):
            qc.i(0)


        #qc.measure(1,0)


        #theta = Parameter('f')

        #qc.rz(theta, 1)
        qc.rz(depth*2*np.pi*f, 0)
        qc.rx(np.pi/2,0)
        #qc.h(0)
        #qc.sx(1)
        return qc
    else:
        qc = QuantumCircuit(2)
        #cr = ClassicalRegister(1, "c")
        #qc.add_register(cr)
        #qc.rx(np.pi/2,1)
        #qc.h(1)
        qc.rx(np.pi / 2, 1)
        #qc.sx(1)

        if initial_state == "0":
            #print(0)
            pass
        elif initial_state == "1":
            #print(1)
            qc.x(0)
        elif initial_state == "+":
            qc.h(0)

        #qc.barrier()
        for d in range(depth):
            #qc.i(0)
            qc.i(1)






        #qc.measure(1,0)


        #theta = Parameter('f')

        #qc.rz(theta, 1)
        qc.rz(depth*2*np.pi*f, 1)

        #qc.rx(np.pi/2,1)
        #qc.h(1)
        qc.rx(np.pi / 2, 1)
        #print(qc)
        #qc.sx(1)
        return qc




def complex_coupling_map(connectivity):

    """
    ASSUMES 8 QUBITS
    Function to boost a ring coupling map to higher connectivities
    can take values 2,4,6,7 and returns a graph where each qubit is connected to 2,4,6 or 7 other qubits


    Parameters
    ----------
    connectivity : int
        required connectivity, one of 2,4,6,7

    Returns
    -------

    """


    if connectivity == 2:
        con_ind = 0
    elif connectivity == 4:
        con_ind = 1
    elif connectivity == 6:
        con_ind = 2
    elif connectivity == 7:
        con_ind = 3

    CM = CouplingMap().from_ring(8)
    print("Initial average connectivity")
    print(len(CM.get_edges()) / 8)

    inds = [0, 1, 2, 3, 4, 5, 6, 7]

    rep_list = [2, 3, 4]

    for j in range(con_ind):
        for i in range(len(inds)):
            # print(inds[i],(inds[i]+rep_list[j])%8)
            CM.add_edge(inds[i], (inds[i] + rep_list[j]) % 8)

        CM.make_symmetric()

    print("Average connectivity")
    non_unique = np.array(CM.get_edges())
    print(len(non_unique) / 8)

    unique_edges = np.unique(non_unique, axis=0)
    if len(non_unique) != len(unique_edges):
        print("there are dubplicate edges!")
        print("something has ogne wrong!!!")

    return CM

