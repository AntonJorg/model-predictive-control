from dataclasses import dataclass

import scipy
import jax
import jax.numpy as jnp

from dtumpc.models import system_dynamics, sensor_deterministic


@dataclass
class LinearSystem:
    A: jnp.ndarray
    B: jnp.ndarray
    G: jnp.ndarray
    C_y: jnp.ndarray
    D: jnp.ndarray
    C: jnp.ndarray


def linearize_system(x_s, u_s, d_s, params):

    A = jax.jacfwd(
        lambda x: system_dynamics(
            x, 
            u_s,
            d_s,
            params,
        )
    )(x_s)

    B = jax.jacfwd(
        lambda u: system_dynamics(
            x_s, 
            u,
            d_s,
            params,
        )
    )(u_s)

    G = jax.jacfwd(
        lambda d: system_dynamics(
            x_s, 
            u_s,
            d,
            params,
        )
    )(d_s)

    key = jax.random.key(0)
    C_y = jax.jacfwd(lambda x: sensor_deterministic(x, params, key))(x_s)

    D = jnp.zeros((4, 2))
    
    # output == height of two first tanks
    C = C_y[:2]

    return LinearSystem(A, B, G, C_y, D, C)


def discretize_system(sys: LinearSystem, T_s: float):
    bdim = sys.B.shape[1]
    block = jnp.hstack([sys.A, sys.B])
    block = jnp.vstack([
        block,
        jnp.zeros((bdim, sys.A.shape[1] + bdim))
    ])

    block_d = scipy.linalg.expm(block * T_s)

    A_d = jnp.asarray(block_d[:4, :4])
    B_d = jnp.asarray(block_d[:4, 4:])

    return LinearSystem(A_d, B_d, sys.G, sys.C_y, sys.D, sys.C)


def augment_system(sys: LinearSystem):
    A = jnp.block([
        [sys.A, sys.G],
        [jnp.zeros((2, 4)), jnp.eye(2)],
    ])

    B = jnp.vstack([sys.B, jnp.zeros((2, 2))])
    C_y = jnp.hstack([sys.C_y, jnp.zeros((4, 2))])
    C = jnp.hstack([sys.C, jnp.zeros((2, 2))])
    G = jnp.vstack([sys.G, jnp.eye(2)])

    return LinearSystem(
        A, B, G, C_y, sys.D, C
    )