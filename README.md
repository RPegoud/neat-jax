<!-- # ***🧬 Neat JAX*** -->
# `neatJax`: Fast NeuroEvolution of Augmenting Topologies 🪸

<center>
    <img src="https://raw.githubusercontent.com/RPegoud/neat-jax/2d8fe31de24a1af26b90cab1722f6803c7d04567/images/Neat%20logo.svg?token=AOPYRH6UJEB6QXS5H26YVX3FZCJ26" width="170" align="right"/>
</center>

<a href= "https://github.com/psf/black">
<img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
<a href="https://github.com/RPegoud/jym/blob/main/LICENSE">
<img src="https://img.shields.io/github/license/RPegoud/jym" /></a>
<a href="https://github.com/astral-sh/ruff">
<img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json"/></a>

JAX implementation of the Neat ``(Evolving Neural Networks through Augmenting Topologies)`` algorithm.

## ***🚀 TO DO***

Forward Pass:

* [x] Add connection weights
* [x] Add individual activation functions
* [x] Add conditional activation of output nodes, return output values
* [ ] Test forward when a single neuron is linked to multiple receivers
* [ ] Test forward pass on larger architectures

Mutation:

* [ ] Add novelty fields to Network dataclass
* [ ] Determine mutation frequency and common practices (crossing, activations, ...)
* [ ] Implement mutation for two simple networks

## ***📝 References***

* [Efficient Evolution of Neural Network Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf), *Kenneth O. Stanley and Risto Miikkulainen, 2001*
