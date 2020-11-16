using MitosisStochasticDiffEq
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

# initial condition
u0 = 1.1

# set true model parameters
par = [-0.2, 0.1, 0.9]

# set of linear parameters Eq.~(2.2)
plin = copy(par)
pest = copy(par)
sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin,dt=dt)

samples1 = MitosisStochasticDiffEq.sample(sdekernel, u0, K, save_noise=false).u
samples2 = [MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=false)[2] for _ in 1:K]

@test isapprox(mean(samples1), mean(samples2), rtol=5e-3)
@test isapprox(cov(samples1), cov(samples2), rtol=5e-3)



# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, 1.5, 0.1];
NT = NamedTuple{mynames}(myvalues)


message, backward = MitosisStochasticDiffEq.backwardfilter(sdekernel, NT)

x0 = 1.34
ll0 = randn()

samples1 = [MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0))[1][1,end] for k in 1:K]
samples2 = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0), numtraj=K)[1][1,end,:]

@test isapprox(mean(samples1), mean(samples2), rtol=5e-3)
@test isapprox(cov(samples1), cov(samples2), rtol=1e-2)
