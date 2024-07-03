import jax
import jax.numpy as jnp

activation_fns_list = [
    lambda x: jnp.float32(x),  # identity
    lambda x: 1 / (1 + jnp.exp(-x)),  # sigmoid
    lambda x: jnp.divide(1, x),  # inverse
    lambda x: jnp.sinh(x) / jnp.cosh(x),  # hyperbolic cosine
    lambda x: jnp.float32(jnp.maximum(0, x)),  # relu
    lambda x: jnp.float32(jnp.abs(x)),  # absolute value
    lambda x: jnp.sin(x),  # sine
    lambda x: jnp.exp(jnp.square(-x)),  # gaussian
    lambda x: jnp.float32(jnp.sign(x)),  # step
]


def get_activation_fn(activation_index: int, x: float) -> jnp.float32:
    """
    Given an index, selects an activation function and computes `activation(x)`.

    ```python
        0: jnp.float32(x) # identity function
        1: 1 / (1 + jnp.exp(-x)),  # sigmoid
        2: jnp.divide(1, x),  # inverse
        3: jnp.sinh(x) / jnp.cosh(x),  # hyperbolic cosine
        4: jnp.float32(jnp.maximum(0, x)),  # relu
        5: jnp.float32(jnp.abs(x)),  # absolute value
        6: jnp.sin(x),  # sine
        7: jnp.exp(jnp.square(-x)),  # gaussian
        8: jnp.float32(jnp.sign(x)),  # step
        ```
    """
    return jax.lax.switch(
        activation_index,
        activation_fns_list,
        operand=x,
    )
