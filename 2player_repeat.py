from graph_tool.all import *
import simul
import utility
import matplotlib.pyplot as plt
import numpy as np

g = Graph()
K = 6
T = 10000
num_simul = 100
v1 = g.add_vertex()
v2 = g.add_vertex()
g.add_edge(v1,v2)
g.add_edge(v2,v1)
R = g.new_vertex_property("object")
V = g.new_vertex_property("object")
g.vp.R = R
g.vp.V = V




correlation_all = np.zeros(T)
MI_all = np.zeros(T)
initial = np.zeros(num_simul)
final = np.zeros(num_simul)

for i in range(num_simul):
    # initialization
    utility.random_initialization(g,K)
    initial[i] = (np.corrcoef(g.vp.V[0], y= g.vp.V[1])[0,1])

    # simulation
    V_record, P_record = simul.associative_diffusion(g, K,  T, lmda = 0.9)
    correlation = np.zeros(T)
    MI = np.zeros(T)
    # calculate correlation and mutual information
    for t in range(T):
        # correlation 
        correlation[t] = np.abs(np.corrcoef(V_record[t,0,:], y= V_record[t,1,:])[0,1])
        
        # mutual information
        # P_b1 = P_record[t].copy()
        # P_b1 = P_b1[:,:, np.newaxis]
        # P_b1 = np.repeat(P_b1, K, axis=-1)
        # P_b2 = P_b1/np.transpose(1-P_b1, (0,2,1))
        # P_b1b2 = P_b2 * np.transpose(P_b1, (0,2,1))
        # eye = np.eye(K)
        # eye = eye[np.newaxis,:,:]
        # eyes = np.repeat(eye, 2, axis=0)
        # P_b1b2 = P_b1b2 * (1-eyes)
        # P_b1_avg = np.average(np.transpose(P_b1, (0,2,1)), axis=0)
        # P_b2_avg = np.average(P_b2, axis=0)
        # P_b1b2_avg = np.average(P_b1b2, axis=0)

        # # masked
        # P_b1_avg = np.ma.masked_where(P_b1_avg == 0, P_b1_avg)
        # P_b2_avg = np.ma.masked_where(P_b2_avg == 0, P_b2_avg)
        # P_b1b2_avg = np.ma.masked_where(P_b1b2_avg == 0, P_b1b2_avg)
        # MI_t = np.ma.masked_invalid(P_b1b2_avg * (np.ma.log(P_b1b2_avg)-np.ma.log(P_b1_avg)-np.ma.log(P_b2_avg))).sum()
        # MI[t] = MI_t

    correlation_all += correlation
    # MI_all += MI
    if i%1 ==0:
        print(i, "rounds")
    # calculate mutual information


    # final 
    final[i] = np.corrcoef(V_record[-1,0,:], y= V_record[-1,1,:])[0,1]

plt.figure()
plt.ylabel("Absolute Correlation")
plt.xlabel("Time")
plt.plot(range(T),correlation_all/num_simul)

plt.figure()
plt.ylabel("Mutual Information")
plt.xlabel("Time")

plt.plot(range(T), MI_all/num_simul)

plt.figure()
plt.ylabel("Final Correltaion")
plt.xlabel("Initial Correlation")
plt.scatter(initial, final)
plt.show()
