# MitosisStochasticDiffEq.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mschauer.github.io/MitosisStochasticDiffEq.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mschauer.github.io/MitosisStochasticDiffEq.jl/dev)
[![Build Status](https://travis-ci.com/mschauer/MitosisStochasticDiffEq.jl.svg?branch=master)](https://travis-ci.com/mschauer/MitosisStochasticDiffEq.jl)


Implement the [Mitosis transformation rules](https://github.com/mschauer/Mitosis.jl) `backwardfilter` and `forwardguiding` for for SciML's [`StochasticDiffEq`](https://github.com/SciML/StochasticDiffEq.jl) problems. 


## Synopsis

MitosisStochasticDiffEq implements the backward filter and the forward change of measure  of the Automatic Backward Filtering Forward Guiding paradigm  (van der Meulen and Schauer, 2020) as transformation rules for SDE models,  suitable to be incorporated into probabilistic programming approaches.

In particular, this package implements the equations ... of section 9.1, [2] further detailed in [1]. The recursion for the quantity c in [1, Theorem 3.3 (Information filter)] is replaced by the simpler rule from [2, Example 10.8.]



## References

* [1] Marcin Mider, Moritz Schauer, Frank van der Meulen (2020): Continuous-discrete smoothing of diffusions. [[arxiv:1712.03807]](https://arxiv.org/abs/arxiv:1712.03807).
* [2] Frank van der Meulen, Moritz Schauer (2020): Automatic Backward Filtering Forward Guiding for Markov processes and graphical models. [[arXiv:2010.03509]](https://arxiv.org/abs/2010.03509).

