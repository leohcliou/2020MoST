import graph_tool.all as gt
import numpy as np
import utility

def associative_diffusion(g, K,T, lmda):
    # g: topology for interaction; graph-tool directed graph
    # T: up
    edges = g.get_edges()
    
    num_edges = edges.shape[0]

    V_record = np.zeros((T, g.num_vertices(), K))
    P_record = np.zeros((T, g.num_vertices(), K))

    for t in range(T):
        e_id = np.random.choice(num_edges)
        A = edges[e_id, 0]
        B = edges[e_id, 1]
        V_A = g.vp.V[A].copy()
        V_B = g.vp.V[B].copy()

        # act
        P = np.exp(V_A)
        P = P/np.sum(P)
        i, j = np.random.choice(K, 2, replace=False, p=P)

        # update associative
        g.vp.R[B][i,j] += 1 
        g.vp.R[B][j,i] += 1 

        # update preference
        V_new_B = V_B.copy()
        # k = i if abs(V_B[i])<abs(V_B[j]) else j
        k = i if np.random.uniform()<1/2 else j
        V_new_B[k] +=  np.random.normal()
        CS = utility.constraint_satisfaction(V_B.copy(), g.vp.R[B].copy())
        CS_new = utility.constraint_satisfaction(V_new_B.copy(), g.vp.R[B].copy())
        g.vp.V[B][k] = V_new_B[k] if CS_new > CS else g.vp.V[B][k]

        # check
        # print("preference A", g.vp.V[A])
        # print("P", P)
        # print("preference B", V_B)
        # print("i, j, k", i, j, k)
        # print("associatition", g.vp.R[B])
        # print("candidate preference B", V_new_B)
        # print("CS old and new", CS, CS_new)
        # print("result B", g.vp.V[B])

        # decay apply
        g.vp.R[B] *= lmda 
        g.vp.R[A] *= lmda 
        

        # record
        for v in g.vertices():
            V_record[t, int(v), :] = g.vp.V[v].copy()
            p = np.exp(g.vp.V[v].copy())
            P_record[t, int(v), :] = p/np.sum(p)

        

    
    return(V_record, P_record)



        

