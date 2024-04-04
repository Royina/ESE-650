import numpy as np
from scipy import io
from quaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import math
import matplotlib.pyplot as plt
import scipy as sp

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

## acceleration calibration
def accel_calib(accel, rot_mat, n_ts):
    actual = np.transpose(rot_mat[:6], (0,2,1)) @ np.array([0,0, 9.81])
    raw_accel = accel[:,:6].T
    actual = (actual * 1023)/3300

    alpha = []
    beta = []

    for i in range(3):
        x = actual[:,i].reshape((actual.shape[0], 1))
        x = np.hstack([x, np.ones((actual.shape[0], 1))])
        y = raw_accel[:, i].reshape((x.shape[0],1))
        alpha_i, beta_i = np.linalg.lstsq(x,y, rcond=None)[0]
        alpha.append(alpha_i)
        beta.append(beta_i)
    beta[2] = (beta[0]+beta[1])/2
    alpha_yaw = 3300/(1023*np.mean(9.81/(accel[2,:100]-beta[2])))
    beta = np.array(beta)
    alpha = alpha_yaw * np.array([1,1,1])
    
    actual_accel = (accel[:,1:n_ts] - beta.reshape((-1,1)))*(3300/(1023*alpha.reshape((-1,1))))
    return alpha, beta, actual_accel

## gyroscope calibration
def gyro_calib(gyro, rot_mat, n_ts, ts):
    R_diff_w = (rot_mat[1:n_ts] - rot_mat[:n_ts-1])/(ts[:,1:n_ts].reshape((-1,1,1)) - ts[:,:n_ts-1].reshape((-1,1,1)))
    R_diff_b = np.transpose(rot_mat[1:n_ts], (0,2,1)) @ R_diff_w 

    wx = (R_diff_b[:,2,1] - R_diff_b[:,1,2])/2
    wy = (R_diff_b[:,0,2] - R_diff_b[:,2,0])/2
    wz = (R_diff_b[:,1,0] - R_diff_b[:,0,1])/2

    wx = np.clip(wx,-2, 2)
    wx = np.convolve(wx, np.ones(5)/5, 'same')

    wy = np.clip(wy, -2, 2)
    wy = np.convolve(wy, np.ones(5)/5, 'same')

    wz = np.clip(wz, -2, 2)
    wz = np.convolve(wz, np.ones(5)/5, 'same')

    w = np.array([wx,wy,wz])


    alpha = []
    beta = []

    for i in range(3):
        a = np.hstack([gyro[i,1:n_ts].reshape((-1,1)), np.ones_like(gyro[i,1:n_ts]).reshape((-1,1))])
        m, c = np.linalg.lstsq(a, w[i] * 180/np.pi, rcond=None)[0]
        alpha.append(3300/(1023*m) *(180/np.pi))
        beta.append(-c/m)

    alpha = np.array(alpha)
    beta = np.array(beta)

    correct_gyro = (gyro[:,1:n_ts] - beta.reshape((-1,1)))*(3300/(1023*alpha.reshape((-1,1))))

    return alpha, beta, correct_gyro

def imu_calibration(accel, gyro, alpha_accel, alpha_gyro, beta_accel, beta_gyro):
    correct_gyro = (gyro - beta_gyro.reshape((-1,1)))*(3300/(1023*alpha_gyro.reshape((-1,1))))
    actual_accel = (accel - beta_accel.reshape((-1,1)))*(3300/(1023*alpha_accel.reshape((-1,1))))

    return actual_accel, correct_gyro

def calibrate_data(accel, gyro):
    # Accelerometer Calibration
    accel_axes = -np.array([accel[0], accel[1], -accel[2]]).T
    accel_bias = np.mean(accel_axes[:20] - np.array([0, 0, 9.81 * 1023 * 59 / 3300]), axis=0)
    accel_sensitivity = 3300 / (1023 * np.array([350, 350, 320]))
    acceleration = (accel_axes - accel_bias) * accel_sensitivity

    # Gyroscope Calibration
    gyro_axes = np.array([gyro[1], gyro[2], gyro[0]]).T
    gyro_bias = np.mean(gyro_axes[:20], axis=0)
    gyro_sens = 3300 / 1023 / np.array([4, 4, 4]) * (np.pi / 180)
    gyro = (gyro_axes - gyro_bias) * gyro_sens

    return acceleration, gyro

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    #vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,1:]
    gyro = imu['vals'][3:6,1:]
    T = np.shape(imu['ts'])[1]-1

    # your code goes here
    # rot_mat = vicon['rots'][:,:,16:]
    # rot_mat = np.transpose(rot_mat, (2,0,1))
    # ts = vicon['ts'][:,16:]
    # n_ts = len(ts[0])
    
    # ## acceleration calibration -- to compute alpha and beta
    # alpha_accel, beta_accel, actual_accel = accel_calib(accel, rot_mat, n_ts)

    # ## gyroscope calibration -- to compute alpha and beta
    # gyro[[0,1,2]] = gyro[[1,2,0]] ## making it wx, wy and wz order
    # alpha_gyro, beta_gyro, actual_gyro = gyro_calib(gyro, rot_mat, n_ts, ts)

    ## calibrate accel and gyro
    # gyro[[0,1,2]] = gyro[[1,2,0]] ## making it wx, wy and wz order
    # accel[:-1,:] = -1 * accel[:-1, :] ## because x and y have negative values
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    alpha_accel = np.array([350, 350, 320])
    alpha_gyro = np.array([4, 4, 4]) * (np.pi / 180)
    beta_accel = accel[:,:20].mean(axis=1)
    beta_gyro = gyro[:,:20].mean(axis=1)
    # actual_accel, actual_gyro = imu_calibration(accel, gyro, alpha_accel, alpha_gyro, beta_accel, beta_gyro)
    actual_accel, actual_gyro = calibrate_data(accel, gyro)
    actual_accel = actual_accel.T
    actual_gyro = actual_gyro.T
    
    # fig, axs = plt.subplots(3)
    # axs[0].plot(accel[0, :])
    # axs[1].plot(accel[1, :])
    # axs[2].plot(accel[2, :])
    # fig.savefig('accel_raw.png')

    fig, axs = plt.subplots(3)
    axs[0].plot(actual_accel[0, :])
    axs[1].plot(actual_accel[1, :])
    axs[2].plot(actual_accel[2, :])
    fig.savefig('corrected_accel.png')

    # fig, axs = plt.subplots(3)
    # axs[0].plot(gyro[0, :])
    # axs[1].plot(gyro[1, :])
    # axs[2].plot(gyro[2, :])
    # fig.savefig('gyro_raw.png')

    fig, axs = plt.subplots(3)
    axs[0].plot(actual_gyro[0, :])
    axs[1].plot(actual_gyro[1, :])
    axs[2].plot(actual_gyro[2, :])
    fig.savefig('corrected_gyro.png')

    # euler_angles = R.from_matrix(rot_mat).as_euler('xyz').T
    # fig, axs = plt.subplots(3)
    # axs[0].plot(euler_angles[0, :])
    # axs[1].plot(euler_angles[1, :])
    # axs[2].plot(euler_angles[2, :])
    # fig.savefig('euler_angles.png')

    roll = []
    pitch = []
    yaw = []

    # ## quaternion testing
    # q1 = Quaternion()
    # q1.from_rotm(rot_mat[0])
    # q2 = Quaternion()
    # q2.from_rotm(rot_mat[1])
    # q = q1*q2

    ## state definition
    q1 = Quaternion()
    q1.normalize()
    ekf_state = (q1, np.array([0,0,0.01]))

    ekf_cov = 0.00001 * np.eye(6)

    imu_ts = np.squeeze(imu['ts'])

    ## iterating for every timestep
    for t in range(len(imu_ts)):
        print(t, '-----------------------------------------------------')
        ## generating sigma points
        ## taking random noise here and not in the process model for covariance
        if t == 0:
            dt = imu_ts[t+1] - imu_ts[t]
        else:
            dt = imu_ts[t] - imu_ts[t-1]

        R_dt = 0.001 * np.eye(6) * dt
        Q = 0.05 * np.eye(6)
        
        n_sqrt_ekf_cov = np.sqrt(6) * sp.linalg.sqrtm(ekf_cov + R_dt)
        n_sqrt_ekf_cov = np.concatenate([n_sqrt_ekf_cov, -1* n_sqrt_ekf_cov], axis=1)
        sigma_points = []
        for i in range(12):
            q1 = Quaternion()
            q1.from_axis_angle(n_sqrt_ekf_cov[:3,i])
            sigma_points.append((ekf_state[0]*q1, ekf_state[1]+n_sqrt_ekf_cov[3:,i]))

        ## propagating sigma points
        prop_sigma_points = []
        for i in range(12):
            q1 = Quaternion()
            q1.from_axis_angle(sigma_points[i][1] * dt)
            prop_sigma_points.append((sigma_points[i][0]*q1, sigma_points[i][1]))
        
        ## Gradient descent for new mean and covariance
        q_dash = Quaternion()
        E = np.zeros((3,12))
        e_dash_thresh = 1e-3
        e_dash = 100
        w_sigma = np.zeros((3,12))
        itr = 0
        max_itr = 1000
        while(e_dash>e_dash_thresh):
            E = np.zeros((3,12))
            itr = itr + 1
            if itr > max_itr:
                break
            for i in range(12):
                ei = prop_sigma_points[i][0] * q_dash.inv()
                ei.normalize()
                E[:, i] = ei.axis_angle()
                if np.round(np.linalg.norm(E[:, i]), 7) == 0:
                    E[:, i] = np.zeros((3,))
                else:
                    E[:, i] = E[:, i]
            e_mean = E.mean(axis=1)
            e_mean = np.where(np.abs(e_mean)<1e-10, 0, e_mean)
            # print(q_dash.q)
            q2 = Quaternion()
            q2.from_axis_angle(e_mean)
            q_dash = q2 * q_dash
            q_dash.normalize()
            # print(q_dash.q)
            e_dash = np.linalg.norm(e_mean)
            # print(e_dash)
        print(itr, '-------------------------------------------------')

        w_sigma = np.zeros((3,12))
        for i in range(12):
            w_sigma[:,i] = prop_sigma_points[i][1]
        u_w = w_sigma.mean(axis=1)
        diff_w = w_sigma - u_w.reshape((-1,1))
        diff_w = np.where(np.abs(diff_w)<1e-10, 0, diff_w)
        error_mat_cov = np.concatenate([E, diff_w], axis=0).reshape((6,12))

        # cov_q = np.mean(np.einsum('ijt , jkt -> ikt',E.reshape((E.shape[0], 1, E.shape[1])), E.reshape((1, E.shape[0], E.shape[1]))), axis=-1)
        
        # cov_w = np.mean(np.einsum('ijt , jkt -> ikt',diff_w.reshape((diff_w.shape[0], 1, diff_w.shape[1])), diff_w.reshape((1, diff_w.shape[0], diff_w.shape[1]))), axis=-1)
        # ekf_cov[:3,:3] = cov_q
        # ekf_cov[-3:, -3:] = cov_w  

        ekf_cov = np.mean(np.einsum('ijt , jkt -> ikt',error_mat_cov.reshape((error_mat_cov.shape[0], 1, error_mat_cov.shape[1])), error_mat_cov.reshape((1, error_mat_cov.shape[0], error_mat_cov.shape[1]))), axis=-1)
        ekf_cov = np.where(np.abs(ekf_cov)<1e-10, 0, ekf_cov)      
        ekf_state = (q_dash, u_w)


        ## measurement model
        g_x = np.zeros((6,12))
        f_x = np.zeros((6,12))
        g_vec = Quaternion(scalar=0, vec = [0,0,9.81])
        for i in range(12):
            f_x[:3, i] = prop_sigma_points[i][0].axis_angle()
            f_x[3:, i] = prop_sigma_points[i][1]
            g_x[:3, i] = (prop_sigma_points[i][0].inv() * g_vec * prop_sigma_points[i][0]).axis_angle()
            g_x[3:, i] = prop_sigma_points[i][1]
        y_hat = g_x.mean(axis=1).reshape((-1,1))
        # y_hat[:3] = y_hat[:3] / np.linalg.norm(y_hat[:3])
        y_hat = np.where(np.abs(y_hat)<1e-10, 0, y_hat)
        cov_yy = Q + np.mean(np.einsum('ijt, jkt -> ikt', (g_x - y_hat).reshape((6,1,12)), (g_x - y_hat).reshape((1,6,12))), axis=-1)
        cov_xy = np.concatenate([E, diff_w], axis=0).reshape((6,1,12))
        cov_xy = np.mean(np.einsum('ijt, jkt -> ikt', cov_xy, (g_x - y_hat).reshape((1,6,12))), axis=-1)
        cov_yy = np.where(np.abs(cov_yy)<1e-10, 0, cov_yy)
        cov_xy = np.where(np.abs(cov_xy)<1e-10, 0, cov_xy)

        ## innovation
        inn_accel = actual_accel[:, t].reshape((-1,1)) - y_hat[:3]#actual_accel[:, t].reshape((-1,1))/np.linalg.norm(actual_accel[:, t]) - y_hat[:3]
        inn_gyro = actual_gyro[:, t].reshape((-1,1)) - y_hat[3:]
        innovation = np.concatenate([inn_accel, inn_gyro], axis=0)
        print(innovation)


        ## update step
        K_star = cov_xy @ np.linalg.inv(cov_yy)
        K_star_inn = K_star @ innovation

        q1 = Quaternion()
        q1.from_axis_angle(K_star_inn[3:].reshape((-1,)))
        u_k1 = ekf_state[0] * q1
        ekf_state = (u_k1, ekf_state[1] + K_star_inn[3:].reshape((-1,)))

        ekf_cov = ekf_cov - K_star @ cov_yy @ K_star.T
    
        euler_angles = ekf_state[0].euler_angles()
        roll.append(euler_angles[0])
        pitch.append(euler_angles[1])
        yaw.append(euler_angles[2])

    roll = np.array(roll)
    pitch = np.array(pitch)
    yaw = np.array(yaw)

    ## plotting the last plot -- comment for gradescope
    # vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    # rot_mat = vicon['rots'][:,:,16:]
    # rot_mat = np.transpose(rot_mat, (2,0,1))
    # euler_angles = R.from_matrix(rot_mat).as_euler('xyz').T
    # ts = vicon['ts'][:,16:]
    # n_ts = len(ts[0])
    # fig, axs = plt.subplots(3)
    # axs[0].plot(roll[:n_ts])
    # axs[0].plot(euler_angles[0, :])
    # axs[1].plot(pitch[:n_ts])
    # axs[1].plot(euler_angles[1, :])
    # axs[2].plot(yaw[:n_ts])
    # axs[2].plot(euler_angles[2, :])
    # fig.savefig('euler_angles_end.png')


    # print('End')

    
    # roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw

if __name__ == "__main__":
    estimate_rot()