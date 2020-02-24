import numpy as np

# def random_initialization(K):
#     R = np.ones((K,K))
#     V = np.random.uniform(-1,1,K)
#     return R, V

def random_initialization(g, K): 
    for v in g.vertices():
        g.vp.R[v] = np.ones((K,K))
        g.vp.V[v] = np.random.uniform(-1,1,K)

def constraint_satisfaction(V, R):
    K = len(V)
    v = np.array(V).reshape((K,1))
    v_rep = np.repeat(v, K, axis = 1)
    Omega = np.abs(v_rep - np.transpose(v_rep))
    Omega_stand = Omega/np.amax(Omega)
    R_stand = R/np.amax(R)
    CS = 1/(K*(K-1))*np.sum(np.abs(R_stand - Omega_stand))
    return CS