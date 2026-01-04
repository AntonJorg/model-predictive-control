import numpy as np
import jax
import jax.numpy as jnp

from dtumpc.models import system_dynamics, pidx


def simulate(
        controller: callable,
        disturbance_model: callable,
        sensor_model: callable,
        params: jnp.ndarray,
        key,
        initial_masses=None, 
        iterations=3600,
    ):
    
    if initial_masses is None:
        initial_masses = jnp.zeros(4)

    states = np.empty((iterations + 1, initial_masses.shape[0]))
    observations = np.empty((iterations, initial_masses.shape[0]))
    flows = np.empty((iterations, initial_masses.shape[0]))

    states[0] = np.asarray(initial_masses)

    for t in range(iterations):
        state = jnp.asarray(states[t])
        # observe the tank heights
        key, subkey = jax.random.split(key)
        observation = sensor_model(state, params, subkey)
        observations[t] = np.asarray(observation)

        # pass into controller
        controls = controller(t, observation)

        # fmin = 0, fmax = 500
        #controls = jnp.clip(controls, 0, 500)

        key, subkey = jax.random.split(key)
        disturbance = disturbance_model(t, state, params, subkey)

        xdot = system_dynamics(state, controls, disturbance, params)

        # record flows F_1 to F_4
        flows[t, :2] = controls
        flows[t, 2:] = disturbance

        states[t + 1] = np.asarray(jnp.clip(state + xdot, 0, None))

    # output the tank heights over time,
    # the inputs (controlled and disturbance),
    # and the observations
    return states[:-1] / params[pidx["area_tank"]], flows, observations


def simulate_state_estimation(
        controller: callable,
        disturbance_model: callable,
        sensor_model: callable,
        estimator,
        initial_estimates,
        steady_state,
        setpoints,
        N_horizon,
        params: jnp.ndarray,
        key,
        initial_masses=None, 
        iterations=3600,
    ):
    
    if initial_masses is None:
        initial_masses = jnp.zeros(4)

    ndims = initial_masses.shape[0]

    states = np.empty((iterations + 2, ndims))

    state_ests = np.empty((iterations + 1, ndims + 2))
    noise_ests = np.empty((iterations + 1, 2))
    P_ests = np.empty((iterations + 1, ndims + 2, ndims + 2))

    observations = np.empty((iterations + 1, ndims))
    flows = np.empty((iterations + 1, ndims))

    states[1] = np.asarray(initial_masses)
    flows[0] = np.zeros(ndims)

    state_ests[0] = initial_estimates[0]
    noise_ests[0] = initial_estimates[1]
    P_ests[0] = initial_estimates[2]

    x_s, u_s, d_s = steady_state


    # convert setpoints to deviation variables and reshape
    
    setpoints = setpoints - (x_s / params[pidx["area_tank"]])[:2]
    
    setpoints = jnp.pad(
        setpoints,
        pad_width=((0, N_horizon), (0, 0)),  # pad N columns on the right
        mode="edge"
    )
    
    setpoints = setpoints.reshape(-1)

    for t in range(1, iterations + 1):
        state = jnp.asarray(states[t])
        
        # observation of system
        key, subkey = jax.random.split(key)
        observation = sensor_model(state, params, subkey)
        observations[t] = np.asarray(observation)

        # deviation variables
        # (all _ests variables are already deviations)
        u_dev = jnp.asarray(flows[t - 1, :2]) - u_s
        y_dev = observation - x_s / params[pidx["area_tank"]]

        # state estimation
        state_est, noise_est, P_est = estimator(
            jnp.asarray(state_ests[t - 1]),
            jnp.asarray(noise_ests[t - 1]),
            u_dev,
            y_dev,
            jnp.asarray(P_ests[t - 1]),
        )
        state_ests[t] = state_est
        noise_ests[t] = noise_est
        P_ests[t] = P_est

        # regulation
        horizon_setpoints = setpoints[2 * t:2 * (t + N_horizon)]
        #print(horizon_setpoints)
        delta_u = controller(t, state_est, noise_est, horizon_setpoints, params)

        if jnp.any(jnp.isnan(delta_u)):
            print(t, state_est, noise_est, horizon_setpoints)
            raise RuntimeError("NaNs")

        controls = jnp.clip(u_s + delta_u, 0, 500)

        # unmeasured disturbance
        key, subkey = jax.random.split(key)
        disturbance = disturbance_model(t, state, params, subkey)

        # compute system dynamics
        xdot = system_dynamics(state, controls, disturbance, params)

        # record flows F_1 to F_4 for plotting
        flows[t, :2] = controls
        flows[t, 2:] = disturbance

        # update state
        states[t + 1] = np.asarray(jnp.clip(state + xdot, 0, None))

    # output the tank heights over time,
    # the inputs (controlled and disturbance),
    # the observations,
    # and the state estimates
    return (
        states[1:-1] / params[pidx["area_tank"]], 
        flows[1:], 
        observations[1:], 
        (state_ests[1:] + np.concatenate([x_s, d_s])) / np.concatenate([params[pidx["area_tank"]], np.ones(2)]),
    )