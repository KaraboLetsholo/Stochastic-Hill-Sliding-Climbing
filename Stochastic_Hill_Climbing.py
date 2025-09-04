import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.lax as lax
import wandb

# --- IGNORE ---
#This is a script for normal stochastic hill climbing,
#it optimizes a given function by iteratively exploring the solution space.
# The optimization process involves generating random perturbations to the current solution
# and accepting or rejecting these perturbations based on their impact on the objective function.

wandb.init()

@jax.jit
def objective(params):
    return 1*jnp.sum((params) ** 2)

obj = lambda params: objective(params)

params = jnp.array([100.0])

distribution = jax.random.uniform(jax.random.PRNGKey(0), shape=(1000,), minval=-1.0, maxval=1.0)
neighbor = params + distribution
new_param_value = jax.random.choice(jax.random.PRNGKey(0), a=neighbor, shape=(), replace=True)

count = 0

while count < 1000:
        if obj(new_param_value) < obj(params):
            params = new_param_value
            distribution = jax.random.uniform(jax.random.PRNGKey(0), shape=(1000,), minval=-1.0, maxval=1.0)
            neighbor = params + distribution
            new_param_value = jax.random.choice(jax.random.PRNGKey(0), a=neighbor, shape=(), replace=True)
            count += 1
            #print(f"Iteration: {count}, Objective: {obj(params)}, Params: {params}")
            wandb.log({"Iteration": count, "Objective": obj(params), "Params": params})
        else:
            distribution = jax.random.uniform(jax.random.PRNGKey(0), shape=(10,), minval=-1.0, maxval=1.0)
            neighbor = params + distribution
            new_param_value = jax.random.choice(jax.random.PRNGKey(0), a=neighbor, shape=(), replace=True)
            count += 1
            #print(f"Iteration: {count}, Objective: {obj(params)}, Params: {params}")

            wandb.log({"Iteration": count, "Objective": obj(params), "Params": params})
