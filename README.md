# MitosisStochasticDiffEq.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mschauer.github.io/MitosisStochasticDiffEq.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mschauer.github.io/MitosisStochasticDiffEq.jl/dev)
[![Build Status](https://github.com/mschauer/MitosisStochasticDiffEq.jl/workflows/CI/badge.svg)](https://github.com/mschauer/MitosisStochasticDiffEq.jl/actions)

Implements the [Mitosis transformation rules](https://github.com/mschauer/Mitosis.jl) `backwardfilter` and `forwardguiding` for for SciML's [`StochasticDiffEq`](https://github.com/SciML/StochasticDiffEq.jl) problems. 

If the (possibly non-linear) drift depends linearly on parameters, estimate the parameters from continuous observations by `regression`. 


## Synopsis

*MitosisStochasticDiffEq* implements the backward filter and the forward change of measure  of the Automatic Backward Filtering Forward Guiding paradigm  (van der Meulen and Schauer, 2020) as transformation rules for SDE models,  suitable to be incorporated into probabilistic programming approaches.

In particular, this package implements the equations ... of section 9.1, [2] further detailed in [1]. The recursion for the quantity c in [1, Theorem 3.3 (Information filter)] is replaced by the simpler rule from [2, Example 10.8.]

## Show reel

### Bayesian regression on the drift parameter of an SDE
```julia
using StochasticDiffEq
using Random
using MitosisStochasticDiffEq
import MitosisStochasticDiffEq as MSDE
using LinearAlgebra, Statistics

# Model and sensitivity
function f(du, u, θ, t)
    c = 0.2 * θ
    du[1] = -0.1 * u[1] + c * u[2]
    du[2] = - c * u[1] - 0.1 * u[2]
    return
end
function g(du, u, θ, t)
    fill!(du, 0.15)
    return
end

# b is linear in the parameter with Jacobian 
function b_jac(J,x,θ,t)
    J .= false
    J[1,1] =   0.2 * x[2] 
    J[2,1] = - 0.2 * x[1]
    nothing
end
# and intercept
function b_icpt(dx,x,θ,t)
    dx .= false
    dx[1] = -0.1 * x[1]
    dx[2] = -0.1 * x[2]
    nothing
end

# Simulate path ensemble 
x0 = [1.0, 1.0]
tspan = (0.0, 20.0)
θ0 = 1.0
dt = 0.05
t = range(tspan...; step=dt)

prob = SDEProblem{true}(f, g, x0, tspan, θ0)
ensembleprob = EnsembleProblem(prob)
ensemblesol = solve(
    ensembleprob, EM(), EnsembleThreads(); dt=dt, saveat=t, trajectories=1000
)

# Inference on drift parameters
sdekernel = MSDE.SDEKernel(f,g,t,0*θ0)
ϕprototype = zeros((length(x0),length(θ0))) # prototypes for vectors
yprototype = zeros((length(x0),))
R = MSDE.Regression!(sdekernel,yprototype,paramjac_prototype=ϕprototype,paramjac=b_jac,intercept=b_icpt)
prior_precision = 0.1I(1)
posterior = MSDE.conjugate(R, ensemblesol, prior_precision)
print(mean(posterior)[], " ± ", sqrt(cov(posterior)[]))
```


## References

* [1] Marcin Mider, Moritz Schauer, Frank van der Meulen (2020): Continuous-discrete smoothing of diffusions. [[arxiv:1712.03807]](https://arxiv.org/abs/arxiv:1712.03807).
* [2] Frank van der Meulen, Moritz Schauer (2020): Automatic Backward Filtering Forward Guiding for Markov processes and graphical models. [[arXiv:2010.03509]](https://arxiv.org/abs/2010.03509).

