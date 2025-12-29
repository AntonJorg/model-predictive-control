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


default_params = ModelParameters(
    area_outlet=jnp.ones(4) * 1.2272,
    area_tank=jnp.ones(4) * 380.1327,
    g=981,
    rho=1,
    gamma1=0.58,
    gamma2=0.72,
)


def system_dynamics(state, controls, disturbance, params):
    # convert mass to height
    tank_heights = jnp.clip(state / (params.area_tank * params.rho), 0, None)
    
    # outflow due to gravity
    outflow = params.area_outlet * jnp.sqrt(2 * params.g * tank_heights)

    # flow from tanks 3 and 4 into tanks 1 and 2
    crossflow = jnp.concatenate((outflow[2:], jnp.zeros(2)))

    # inflow is then pump flow + crossflow
    pump_force = jnp.concatenate((controls, controls[::-1]))    
    inflow = params.flow_rates * pump_force + crossflow

    # change in tank volume equals inflow minus outflow
    xdot = inflow - outflow

    # add disturbance
    disturbance = jnp.concatenate((jnp.zeros(2), disturbance))
    xdot = xdot + disturbance

    # convert to masses and return
    return xdot * params.rho


def sensor_deterministic(state, params):
    return state / params.area_tank


def sensor_stochastic(key, std, state, params):
    key, sub = jax.random.split(key)
    noise = jax.random.normal(sub, shape=(4,))
    return sensor_deterministic(state, params) + noise * std



