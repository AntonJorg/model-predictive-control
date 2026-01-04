import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP


_solver = BoxOSQP()


def solve_qp(
        H: jnp.ndarray,
        g: jnp.ndarray,
        x_min: jnp.ndarray,
        x_max: jnp.ndarray,
        A: jnp.ndarray,
        Ax_min: jnp.ndarray,
        Ax_max: jnp.ndarray,
    ) -> jnp.ndarray:

    A = jnp.vstack([jnp.eye(A.shape[1]), A])

    l = jnp.concatenate([x_min, Ax_min])
    u = jnp.concatenate([x_max, Ax_max])

    x_star = _solver.run(
        params_obj=(H, g), 
        params_eq=A, 
        params_ineq=(l, u)
    )

    return x_star.params.primal[0]


if __name__ == "__main__":
    # Quadratic cost: 0.5 x^T H x + g^T x
    H = jnp.array([[2.0, 0.0],
                [0.0, 2.0]])
    g = jnp.array([-2.0, -5.0])

    # Box constraints on x
    x_min = jnp.array([0.0, 0.0])
    x_max = jnp.array([2.0, 3.0])

    # General linear constraint:  x1 + x2 ∈ [1, 2]
    A = jnp.array([[1.0, 1.0]])
    Ax_min = jnp.array([1.0])
    Ax_max = jnp.array([2.0])

    # Solve
    x_star = solve_qp(
        H,
        g,
        x_min,
        x_max,
        A,
        Ax_min,
        Ax_max,
    )

    print(x_star)