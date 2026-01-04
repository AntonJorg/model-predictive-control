import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_results(
        tank_heights,
        inputs, 
        title=None,
        observations = None,
        targets = None,
        estimations = None,
        fig_path = None,
        file_name = None,
    ):

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    upper_bound_top_row = jnp.max(tank_heights[:,2:]) * 1.2
    upper_bound_low_row = jnp.max(tank_heights[:,:2]) * 1.2

    if title is not None:
        fig.suptitle(title)

    alpha = 0.5

    ax = axes[1, 1]
    ax.set_title("Tank 1")
    ax.plot(tank_heights[:, 0], alpha=alpha, label="True levels")
    if observations is not None:
        ax.plot(observations[:, 0], alpha=alpha, label="Observations")
    if estimations is not None:
        ax.plot(estimations[:, 0], label="Estimated levels")
    if targets is not None:
        ax.plot(targets[:, 0], color="k", linestyle="--", label="Target levels")
    ax.set_ylim(0, upper_bound_low_row)
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    ncol = len(labels) 
    
    ax.legend(
        loc="lower center", 
        ncol=ncol, 
        bbox_to_anchor=(0.5, 0.0), 
        bbox_transform=fig.transFigure
    )

    ax = axes[1, 2]
    ax.set_title("Tank 2")
    ax.plot(tank_heights[:, 1], alpha=alpha)
    if observations is not None:
        ax.plot(observations[:, 1], alpha=alpha)
    if estimations is not None:
        ax.plot(estimations[:, 1])
    if targets is not None:
        ax.plot(targets[:, 1], color="k", linestyle="--")
    ax.set_ylim(0, upper_bound_low_row)
    ax.grid(True)

    ax = axes[0, 1]
    ax.set_title("Tank 3")
    ax.plot(tank_heights[:, 2], alpha=alpha)
    if observations is not None:
        ax.plot(observations[:, 2], alpha=alpha)
    if estimations is not None:
        ax.plot(estimations[:, 2])
    ax.set_ylim(0, upper_bound_top_row)
    ax.grid(True)

    ax = axes[0, 2]
    ax.set_title("Tank 4")
    ax.plot(tank_heights[:, 3], alpha=alpha)
    if observations is not None:
        ax.plot(observations[:, 3], alpha=alpha)
    if estimations is not None:
        ax.plot(estimations[:, 3])
    ax.set_ylim(0, upper_bound_top_row)
    ax.grid(True)

    ax = axes[1, 0]
    ax.set_title("Flow 1 (Controlled)")
    ax.plot(inputs[:, 0])
    ax.set_ylim(0, 500)
    ax.grid(True)

    ax = axes[1, 3]
    ax.set_title("Flow 2 (Controlled)")
    ax.plot(inputs[:, 1])
    ax.set_ylim(0, 500)
    ax.grid(True)

    ax = axes[0, 0]
    ax.set_title("Flow 3 (Disturbance)")
    ax.plot(inputs[:, 2])
    ax.set_ylim(0, 500)
    ax.grid(True)

    ax = axes[0, 3]
    ax.set_title("Flow 4 (Disturbance)")
    ax.plot(inputs[:, 3])
    ax.set_ylim(0, 500)
    ax.grid(True)

    #plt.tight_layout()
    fig.subplots_adjust(bottom=0.10)
    fig.subplots_adjust(hspace=0.3)

    if file_name is not None:
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_markov_params(H, title=None, filename=None):
    n_params = H.shape[0] - 1

    maxval = jnp.max(H)
    margin = 0.05 * maxval
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    
    ax = axes[0, 0]
    ax.plot(jnp.arange(1, n_params + 1), H[1:, 0, 0])
    ax.set_title("$u_1 \\to z_1$")
    ax.grid()
    ax.set_ylim(-margin, maxval + margin)

    ax = axes[0, 1]
    ax.plot(jnp.arange(1, n_params + 1), H[1:, 0, 1])
    ax.set_title("$u_2 \\to z_1$")
    ax.grid()
    ax.set_ylim(-margin, maxval + margin)

    ax = axes[1, 0]
    ax.plot(jnp.arange(1, n_params + 1), H[1:, 1, 0])
    ax.set_title("$u_1 \\to z_2$")
    ax.grid()
    ax.set_ylim(-margin, maxval + margin)

    ax = axes[1, 1]
    ax.plot(jnp.arange(1, n_params + 1), H[1:, 1, 1])
    ax.set_title("$u_2 \\to z_2$")
    ax.grid()
    ax.set_ylim(-margin, maxval + margin)

    if title is not None:
        fig.suptitle(title)

    if filename is not None:
        plt.savefig(filename)

    plt.show()