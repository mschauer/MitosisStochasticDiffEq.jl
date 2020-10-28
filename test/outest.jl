#using Revise
using Mitosis, MitosisStochasticDiffEq, Statistics
using LinearAlgebra, Test
using Random
Random.seed!(124)
using Mitosis: AffineMap

# set true model parameters
par = [-0.1, 0.0, 0.9]

# Samples
K = 800

# define SDE function
f(u,p,t) = p[1]*u .+ p[2]
g(u,p,t) = p[3] .+ 0u # needs to preserve type of u for forwardguided

# time span
tstart = 0.0
tend = 1.0
dt = 0.02

# initial condition
u0 = 1.1

# set of linear parameters Eq.~(2.2)
plin = copy(par)
pest = copy(par)
sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin,p=par,dt=dt)

plin2 = [-0.05, 0.07, 1.0]
sdekernel2 = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin2,p=par,dt=dt)



sol, y_ = MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=true)
y = vcat(y_)
samples_ = [MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=true)[2] for k in 1:K]
samples = vcat.(samples_)

# Compute transition density. See Ludvig Arnold (ISBN 9780486482361 ), also Proposition 3.5. in [1]
B = fill(par[1], 1, 1)
Φ = exp(B*(tend - tstart))
β = fill(0.0, 1)
Σ = fill(par[3]^2, 1, 1)
Λ = lyap(B, Σ)
Q = Λ - Φ*Λ*Φ'


gkernel = Mitosis.kernel(Gaussian; μ=AffineMap(Φ, β), Σ=ConstantMap(Q))

p = gkernel([u0])
p̂ = Gaussian{(:μ, :Σ)}(mean(samples), cov(samples))
@test norm(p.μ - p̂.μ) < 5/sqrt(K)
@test norm(p.Σ - p̂.Σ) < 5/sqrt(K)



m, p = Mitosis.backwardfilter(gkernel, y)
# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, y_, 0.0]; # start with observation μ=y with uncertainty Σ=0
NT = NamedTuple{mynames}(myvalues)

solend, message = MitosisStochasticDiffEq.backwardfilter(sdekernel, NT)

@testset "Mitosis backward" begin
    @test (p.c)[] ≈ solend[3]
    @test (p.Γ\p.F)[] ≈ solend[1] atol=0.02
    @test inv(p.Γ)[] ≈ solend[2] atol=0.02
end

# Try forward
m, p = Mitosis.backwardfilter(gkernel, WGaussian{(:F, :Γ, :c)}(y*10.0, 10.0I, 0.0))
kᵒ = Mitosis.left′(BFFG(), gkernel, m)

pT = kᵒ([u0])

# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, y_, 0.1]; # start with observation μ=y with uncertainty Σ=0.1
NT = NamedTuple{mynames}(myvalues)
solend, message = MitosisStochasticDiffEq.backwardfilter(sdekernel, NT)

solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (u0, 0.0), Z=nothing; save_noise=true)

@test ll == 0

samples = [MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (u0, 0.0), Z=nothing; save_noise=true)[1][1,end] for k in 1:K]

@testset "Mitosis forward" begin
    @test pT.μ[] ≈ mean(samples) atol=0.02
    @test pT.Σ[] ≈ cov(samples) atol=0.02

end


# try tilted forward

solend, message = MitosisStochasticDiffEq.backwardfilter(sdekernel2, NT)
solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel2, message, (u0, 0.0), Z=nothing; save_noise=true)

we((x,c)) = x*exp(c)
samples = [we(MitosisStochasticDiffEq.forwardguiding(sdekernel2, message, (u0, 0.0), Z=nothing; save_noise=true)[1][:,end]) for k in 1:K]

@show std(samples)
ptrue = Mitosis.density(p, [u0])
p̃ = Mitosis.density(WGaussian{(:F,:Γ,:c)}(solend[2]\solend[1], inv(solend[2]), solend[3]), u0)
@testset "Mitosis tilted forward" begin
    @test pT.μ[] ≈ mean(samples)*p̃/ptrue atol=0.02
end
