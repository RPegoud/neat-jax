<!-- # ***🧬 Neat JAX*** -->
# `neatJax`: Fast NeuroEvolution of Augmenting Topologies 🪸

[![Issues](https://img.shields.io/github/issues/RPegoud/neat-jax)](https://github.com/RPegoud/neat-jax/issues)
[![Issues](https://github.com/RPegoud/neat-jax/actions/workflows/lint_and_test.yaml/badge.svg)](https://github.com/RPegoud/neat-jax/actions/workflows/lint_and_test.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<img src="https://raw.githubusercontent.com/RPegoud/neat-jax/2d8fe31de24a1af26b90cab1722f6803c7d04567/images/Neat%20logo.svg?token=AOPYRH6UJEB6QXS5H26YVX3FZCJ26" width="170" align="right"/>

JAX implementation of the Neat ``(Evolving Neural Networks through Augmenting Topologies)`` algorithm.

## ***🚀 TO DO***

Forward Pass:

* [x] Add connection weights
* [x] Add individual activation functions
* [x] Add conditional activation of output nodes, return output values
* [x] Test forward when a single neuron is linked to multiple receivers
* [x] Test forward pass on larger architectures

Mutations:

* [x] Determine mutation frequency and common practices
* [ ] Implement the following mutations:
  * [x] Weight shift
  * [x] Weight reset
  * [x] Add node
  * [x] Add connection
  * [ ] Mutate activation
* [ ] Wrap all mutations in a single function

Misc:

* [ ] Add bias
* [ ] Refactor activation state (maybe include it in the network class)
* [ ] Separate ``max_nodes`` and ``max_connections``
* [ ] Set the minimum sender index to 1 instead of 0
* [ ] Add Hydra config for constant attributes

Crossing:

* [ ] Add novelty fields to Network dataclass
* [ ] Implement crossing for two simple networks
* [ ] Create a Species dataclass
* [ ] Define a distance metrics between networks to cluster species

## ***📝 References***

* [Efficient Evolution of Neural Network Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf), *Kenneth O. Stanley and Risto Miikkulainen, 2001*
