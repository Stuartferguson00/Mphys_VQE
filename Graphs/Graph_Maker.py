import networkx as nx
import pickle
import matplotlib.pyplot as plt
plt.style.use('default')







def save_object(obj, filename):
    #saves pythonobject as pkl
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


#make 1000 Erdos Renyi graphs of a given size with probability of connection = 0.5
size = 10
Graphs = []
for i in range(1000):
    G= nx.erdos_renyi_graph(size,0.5)
    Graphs.append(G)
save_object(Graphs, "Graphs_10.pkl")