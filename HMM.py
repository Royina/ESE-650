import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    def forward(self):

        alpha = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        for i in range(alpha.shape[0]):
            if i == 0:
                alpha[i,:] = self.Initial_distribution * self.Emission[:,self.Observations[i]].T

            else:
                alpha[i, :] = self.Emission[:,self.Observations[i]].T * (alpha[i-1] @ self.Transition)

        return alpha

    def backward(self):

        beta = np.ones((self.Observations.shape[0], self.Transition.shape[0]))
        for i in range(beta.shape[0]-2, -1, -1):
            beta[i, :] = (self.Transition @ (beta[i+1].T * self.Emission[:,self.Observations[i+1]])).T

        return beta

    def gamma_comp(self, alpha, beta):

        gamma =  np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        gamma = alpha * beta 
        gamma = gamma / np.sum(alpha[-1,:])

        return gamma

    def xi_comp(self, alpha, beta, gamma):

        xi = np.zeros((self.Observations.shape[0]-1, self.Transition.shape[0], self.Transition.shape[0]))
        for i in range(self.Observations.shape[0]-1):
            for j in range(self.Transition.shape[0]):
                for k in range(self.Transition.shape[0]):
                    xi[i,j,k] = alpha[i,j] * self.Transition[j,k] * self.Emission[k, self.Observations[i+1]] * beta[i+1,k]

        xi = xi / np.sum(xi, axis=(1,2)).reshape((xi.shape[0], 1, 1))

        return xi

    def update(self, alpha, beta, gamma, xi):

        new_init_state = gamma[0]
        T_prime = (xi.sum(axis=0).T / gamma[:-1,:].sum(axis=0)).T
        M_prime = np.zeros_like(self.Emission)
        for i in range(M_prime.shape[1]):
            indices = np.argwhere(self.Observations==i)
            gamma_sub = gamma[indices]
            M_prime[:,i] = gamma_sub.sum(axis=0) / gamma.sum(axis=0)


        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        P_original = 0
        P_prime = 0

        P_original = np.sum(alpha[-1,:])

        self.Initial_distribution = new_init_state
        self.Transition = T_prime
        self.Emission = M_prime

        alpha_prime = self.forward()

        P_prime = np.sum(alpha_prime[-1,:])
        

        return P_original, P_prime
