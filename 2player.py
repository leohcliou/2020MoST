from graph_tool.all import *
import simul
import utility
import matplotlib.pyplot as plt
import numpy as np

g = Graph()
K = 6
T = 1000
v1 = g.add_vertex()
v2 = g.add_vertex()
g.add_edge(v1,v2)
g.add_edge(v2,v1)
R = g.new_vertex_property("object")
V = g.new_vertex_property("object")
g.vp.R = R
g.vp.V = V
utility.random_initialization(g,K)
V_record = simul.associative_diffusion(g, K,  T, lmda = 0.9)
correlation = np.zeros(T)
for t in range(T):
    correlation[t] = (np.corrcoef(V_record[t,0,:], y= V_record[t,1,:])[0,1])
plt.plot(range(T),correlation)
plt.show()
