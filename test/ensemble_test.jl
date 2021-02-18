using MitosisStochasticDiffEq
using Mitosis
using Test, Random
using Statistics

K = 100_000
Random.seed!(1234)

# define SDE function
f(u,p,t) = p[1]*u .+ p[2]
g(u,p,t) = p[3]

# time span
tstart = 0.0
tend = 1.0
dt = 0.01
trange = tstart:dt:tend

# initial condition
u0 = 1.1

# set true model parameters
par = [-0.2, 0.1, 0.9]

# set of linear parameters Eq.~(2.2)
plin = copy(par)
sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,par)

samples1 = MitosisStochasticDiffEq.sample(sdekernel, u0, K, save_noise=false).u
samples2 = [MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=false)[2] for _ in 1:K]

@test isapprox(mean(samples1), mean(samples2), rtol=1e-2)
@test isapprox(cov(samples1), cov(samples2), rtol=1e-2)


# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, 1.5, 0.1];
NT = NamedTuple{mynames}(myvalues)

sdetildekernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,plin)
message, backward = MitosisStochasticDiffEq.backwardfilter(sdetildekernel, NT)

sdetildekernel2 = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
message2, backward2 = MitosisStochasticDiffEq.backwardfilter(sdetildekernel2, NT)

@test isapprox(backward, backward2, rtol=1e-15)
@test isapprox(Array(message.sol), Array(message2.sol), rtol=1e-15)

x0 = 1.34
ll0 = randn()

samples1 = [MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0))[1][1,end] for k in 1:K]
samples2 = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0), numtraj=K)[1][1,end,:]

@test isapprox(mean(samples1), mean(samples2), rtol=1e-2)
@test isapprox(cov(samples1), cov(samples2), rtol=1e-2)
