using MitosisStochasticDiffEq
using Test, Random
using Statistics

K = 10000
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

@test isapprox(mean(samples1), mean(samples2), rtol=1e-2)
@test isapprox(std(samples1), std(samples2), rtol=1e-3)
