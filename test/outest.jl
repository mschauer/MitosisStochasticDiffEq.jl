#using Revise
using Mitosis, MitosisStochasticDiffEq, Statistics
using LinearAlgebra, Test

using Mitosis: AffineMap

# set true model parameters
p = [-0.1, 0.0, 0.9]

# Samples
K = 400

# define SDE function
f(u,p,t) = p[1]*u + p[2]
g(u,p,t) = p[3]

# time span
tstart = 0.0
tend = 1.0
dt = 0.02

# intial condition
u0 = 1.1

# set of linear parameters Eq.~(2.2)
plin = copy(p)
pest = [2.0]
sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin,p=p,dt=dt)
sol, solend = MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=true)
samples_ = [MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=true)[2] for k in 1:K]
samples = vcat.(samples_)

B = fill(p[1], 1, 1)
Φ = exp(B*(tend - tstart))
β = fill(0.0, 1)
Σ = fill(p[3]^2, 1, 1)
Λ = lyap(B, Σ)
Q = Λ - Φ*Λ*Φ'


gkernel = kernel(Gaussian; μ=AffineMap(Φ, β), Σ=ConstantMap(Q))

p = gkernel([u0])
p̂ = Gaussian{(:μ, :Σ)}(mean(samples), cov(samples))
@test norm(p.μ - p̂.μ) < 5/sqrt(K)
@test norm(p.Σ - p̂.Σ) < 5/sqrt(K)
