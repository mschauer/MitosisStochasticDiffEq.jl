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

# initial condition
u0 = 1.1

# set of linear parameters Eq.~(2.2)
plin = copy(p)
pest = [2.0]
sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin,p=p,dt=dt)
sol, y_ = MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=true)
y = vcat(y_)
samples_ = [MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=true)[2] for k in 1:K]
samples = vcat.(samples_)

# Compute transition density. See Ludvig Arnold (ISBN 9780486482361 ), also Proposition 3.5. in [1]
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



m, p = backwardfilter(gkernel, y)
# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, y_, 0.0];
NT = NamedTuple{mynames}(myvalues)

solend, message = MitosisStochasticDiffEq.backwardfilter(sdekernel, NT)

@testset "Mitosis" begin
    @test (p.c)[] ≈ solend[3]
    @test (p.Γ\p.F)[] ≈ solend[1] atol=0.02
    @test inv(p.Γ)[] ≈ solend[2] atol=0.02
end
