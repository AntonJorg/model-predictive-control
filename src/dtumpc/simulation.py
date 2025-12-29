import jax.numpy as jnp

from dtumpc.models import system_dynamics, ModelParameters


def simulate(
        controller: callable,
        disturbance_model: callable,
        sensor_model: callable,
        params: ModelParameters,
        initial_masses=None, 
        iterations=3600,
    ):
    
    if initial_masses is None:
        initial_masses = jnp.zeros(4)

    states = jnp.empty((iterations + 1, initial_masses.shape[0]))
    observations = jnp.empty((iterations, initial_masses.shape[0]))
    flows = jnp.empty((iterations, initial_masses.shape[0]))

    states[0] = initial_masses

    for t in range(iterations):
        # observe the tank heights
        observation = sensor_model(states[t], params)
        observations[t] = observation

        # pass into controller
        controls = controller(t, observation)

        # fmin = 0, fmax = 500
        #controls = jnp.clip(controls, 0, 500)

        disturbance = disturbance_model(t, states[t])

        xdot = system_dynamics(states[t], controls, disturbance, params)

        # record flows F_1 to F_4
        flows[t, :2] = controls
        flows[t, 2:] = disturbance

        states[t + 1] = jnp.clip(states[t] + xdot, 0, None)

    # output the tank heights over time,
    # the inputs (controlled and disturbance),
    # and the observations
    return states[:-1] / params.area_tank, flows, observations