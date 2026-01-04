from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ModelParameters:
    area_outlet: jnp.ndarray
    area_tank: jnp.ndarray
    g: float
    rho: float
    gamma1: float
    gamma2: float

    def __post_init__(self):
        flow_rates = jnp.array([
            self.gamma1,
            self.gamma2,
            1 - self.gamma2,
            1 - self.gamma1,
        ])

        object.__setattr__(self, "flow_rates", flow_rates)


def construct_params(gamma1, gamma2):
    flow_rates = jnp.array([
        gamma1,
        gamma2,
        1 - gamma2,
        1 - gamma1,
    ])

    area_outlet=jnp.ones(4) * 1.2272
    area_tank=jnp.ones(4) * 380.1327

    scalars = jnp.array([gamma1, gamma2, 981.0, 1.0])

    return jnp.vstack([area_outlet, area_tank, flow_rates, scalars])


default_params = construct_params(0.58, 0.72)


pidx = {
    "area_outlet": 0,
    "area_tank": 1,
    "flow_rates": 2,
    "gamma1": (3, 0),
    "gamma2": (3, 1),
    "g": (3, 2),
    "rho": (3, 3),
}


@jax.jit
def system_dynamics(state, controls, disturbance, params):

    # convert mass to height
    tank_heights = jnp.clip(state / (params[pidx["area_tank"]] * params[pidx["rho"]]), 0, None)
    
    # outflow due to gravity
    outflow = params[pidx["area_outlet"]] * jnp.sqrt(2 * params[pidx["g"]] * tank_heights)

    # flow from tanks 3 and 4 into tanks 1 and 2
    crossflow = jnp.concatenate((outflow[2:], jnp.zeros(2)))

    # inflow is then pump flow + crossflow
    pump_force = jnp.concatenate((controls, controls[::-1]))    
    inflow = params[pidx["flow_rates"]] * pump_force + crossflow

    # change in tank volume equals inflow minus outflow
    xdot = inflow - outflow

    # add disturbance
    disturbance = jnp.concatenate((jnp.zeros(2), disturbance))
    xdot = xdot + disturbance

    # convert to masses and return
    return xdot * params[pidx["rho"]]


def sensor_deterministic(state, params, key):
    return state / params[pidx["area_tank"]]


def sensor_stochastic(state, params, key, std):
    noise = jax.random.normal(key, shape=(4,))
    return state / params[pidx["area_tank"]] + noise * std


def disturbance_constant(t, state, params, key, value):
    return value

def disturbance_piecewise_constant(t, state, params, key, values, scale):
    return values[t // scale]

def disturbance_sde(t, state, params, key, mean, cov):
    return jax.random.multivariate_normal(key, mean, cov)
