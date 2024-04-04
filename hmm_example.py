import numpy as np
import matplotlib.pyplot as plt
from HMM import HMM
import random
import pandas as pd


if __name__ == "__main__":

    Observations = np.array(['null', 'LA', 'LA', 'null', 'NY', 'null', 'NY', 'NY', 'NY', 'null', 'NY', 'NY', 'NY', 'NY', 'NY', 'null', 'null', 'LA', 'LA', 'NY'])
    Observations = np.where(Observations=='LA', 0, np.where(Observations=='NY', 1, 2))
    Initial_distribution = np.zeros((2,))
    Initial_distribution[:] = 0.5

    Transition = np.zeros((2,2))
    Transition[:,:] = 0.5

    Emission = np.array([[0.4, 0.1, 0.5], [0.1,0.5,0.4]])

    hmm = HMM(Observations, Transition, Emission, Initial_distribution)

    alpha = hmm.forward()
    beta = hmm.backward()
    gamma = hmm.gamma_comp(alpha, beta)

    df = pd.DataFrame(columns=['gamma_LA', 'gamma_NY', 'alpha_LA', 'alpha_NY', 'beta_LA', 'beta_NY'])
    df['gamma_LA'] = gamma[:, 0]
    df['gamma_NY'] = gamma[:, 1]
    df['alpha_LA'] = alpha[:, 0]
    df['alpha_NY'] = alpha[:, 1]
    df['beta_LA'] = beta[:, 0]
    df['beta_NY'] = beta[:, 1]

    seq_state = np.where(gamma.argmax(axis=1)==0,'LA','NY')

    print(df)
    print('Point-wise most likely sequence of states is :')
    print(seq_state)

    print('Old Initial Distribution:')
    print(Initial_distribution)
    print('Old Transition Matrix:')
    print(Transition)
    print('Old Observation Matrix:')
    print(Emission)

    xi = hmm.xi_comp(alpha, beta, gamma)
    T_prime, M_prime, new_init_state = hmm.update(alpha, beta, gamma, xi)

    print('New Initial Distribution:')
    print(new_init_state)
    print('New Transition Matrix:')
    print(T_prime)
    print('New Observation Matrix:')
    print(M_prime)

    P_original, P_prime = hmm.trajectory_probability(alpha, beta, T_prime, M_prime, new_init_state)

    print('P_original : ', P_original)
    print('P_prime : ', P_prime)

