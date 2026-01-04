import numpy as np
import jax.numpy as jnp

from dtumpc.linearize import LinearSystem

def markov_parameters(A, B, C, horizon):
    # H_0 hardcoded as 0, since D=0 in all models considered

    H = np.zeros((horizon + 1, C.shape[0], B.shape[1]))

    Ai = jnp.eye(A.shape[0])

    for i in range(horizon):
        H[i + 1] = np.asarray(C @ Ai @ B)
        Ai = Ai @ A

    return H


def kalman_filter_stationary(x_prev, w_prev, u_prev, y, P, sys: LinearSystem, Rvv, Rwv):
    # gains
    Re = sys.C_y @ P @ sys.C_y.T + Rvv
    Re_inv = jnp.linalg.inv(Re)

    Kfx = P @ sys.C_y.T @ Re_inv
    Kfw = Rwv @ Re_inv
    
    #prediction
    x_pred = sys.A @ x_prev + sys.B @ u_prev + sys.G @ w_prev
    
    y_pred = sys.C_y @ x_pred

    error = y - y_pred

    # filtering
    x_out = x_pred + Kfx @ error
    w_out = Kfw @ error

    return x_out, w_out, P


def kalman_filter_dynamic(x_prev, w_prev, u_prev, y, P_prev, sys: LinearSystem, Rvv, Rwv, Q):
    # prediction
    x_pred = sys.A @ x_prev + sys.B @ u_prev + sys.G @ w_prev
    P_pred = sys.A @ P_prev @ sys.A.T + Q

    # innovation / gains
    Re = sys.C_y @ P_pred @ sys.C_y.T + Rvv
    Re_inv = jnp.linalg.inv(Re)

    Kfx = P_pred @ sys.C_y.T @ Re_inv
    Kfw = Rwv @ Re_inv

    # measurement update
    y_pred = sys.C_y @ x_pred
    error = y - y_pred

    x_out = x_pred + Kfx @ error
    w_out = Kfw @ error

    I = jnp.eye(P_prev.shape[0])
    KC = Kfx @ sys.C_y
    P_out = (I - KC) @ P_pred @ (I - KC).T + Kfx @ Rvv @ Kfx.T  # Joseph form

    return x_out, w_out, P_out


def kalman_predictions_stationary(sys: LinearSystem, x):
    pass