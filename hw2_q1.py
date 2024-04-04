import numpy as np
import copy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x0 = np.random.normal(1,2)
    samples = 100

    ## creating the dataset
    x = copy.deepcopy(x0)
    a=-1
    y_obs = []
    
    R = np.array([[1e-4, 0], [0, 1]])
    for i in range(samples):
        epsilon = np.random.normal(0,1)
        vk = np.random.normal(0,0.5)
        x = a*x + epsilon
        y_obs.append(np.sqrt((x**2) + 1)+vk)
        

    ## initialising the ekf
    y_val = []
    ekf = np.array([-2, x0]).reshape((2,1))
    a_ests = []
    cov = np.array([[1e-4,0],[0, 4]])
    cov_a = []

    ## ekf
    for i in range(samples):
        epsilon = np.random.normal(0,1,1)[0]
        vk = np.random.normal(0,0.5,1)[0]
        y_val.append(np.sqrt((ekf[1][0]**2) + 1)+vk)
        a = ekf[0][0]
        x = ekf[0][0]*ekf[1][0] + epsilon

        
        A = np.array([[1, 0],[ekf[1][0], ekf[0][0]]]) ## f(a) and f(n) derivatives wrt a and x
        
        cov = A @ cov @ A.T + R

        C = np.array([0, x/(np.sqrt((x**2)+1))]).reshape((1,2))

        K = (cov @ C.T) / (C @ cov @ C.T + 0.9)
        ekf = np.array([a,x]).reshape((2,1)) + (K * (y_obs[i] - (np.sqrt((x**2) + 1)+vk)))
        cov = (np.eye(2) - K @ C) @ cov
        cov_a.append(cov[0][0])
        a_ests.append(ekf[0][0])
        print(ekf[0][0])

print('final a:', a_ests[-1])

plt.plot(-1*np.ones_like(np.array(a_ests)), label='ground truth')
plt.plot(np.array(a_ests), label='mean')
plt.plot(np.array(a_ests)+np.sqrt(np.array(cov_a)), label='mean+std')
plt.plot(np.array(a_ests)-1*np.sqrt(np.array(cov_a)), label = 'mean-std')
plt.legend()
plt.savefig('hw2_q1_'+str(samples)+'_plot.png')


    
    
        