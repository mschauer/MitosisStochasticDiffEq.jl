#using Revise
using Mitosis, MitosisStochasticDiffEq, Statistics
using LinearAlgebra, Test
using Random
Random.seed!(126)
using Mitosis: AffineMap
atol = 0.015
# set true model parameters
par = [-0.2, 0.1, 0.9]

# Samples
K = 1000

# define SDE function
f(u,p,t) = p[1]*u .+ p[2]
g(u,p,t) = p[3] .+ 0u # needs to preserve type of u for forwardguided

# time span
tstart = 0.0
tend = 1.0
dt = 0.01

# initial condition
u0 = 1.1

# set of linear parameters Eq.~(2.2)
plin = copy(par)
pest = copy(par)
sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin,dt=dt)

plin2 = par2 = [-0.1, -0.05, 1.0]
sdekernel2 = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin2,dt=dt)



sol, y_ = MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=true)

y_ = 1.39
y = vcat(y_)
samples_ = [MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=true)[2] for k in 1:K]
samples = vcat.(samples_)

# Compute transition density. See Ludvig Arnold (ISBN 9780486482361 ), also Proposition 3.5. in [1]
B = fill(par[1], 1, 1)
Φ = exp(B*(tend - tstart))
β = (Φ - I)*inv(B)*fill(par[2], 1)
Σ = fill(par[3]^2, 1, 1)
Λ = lyap(B, Σ)
Q = Λ - Φ*Λ*Φ'

gkernel = Mitosis.kernel(Gaussian; μ=AffineMap(Φ, β), Σ=ConstantMap(Q))

B̃ = fill(par2[1], 1, 1)
Φ̃ = exp(B̃*(tend - tstart))
β̃ = (Φ̃ - I)*inv(B̃)*fill(par2[2], 1)
Σ̃ = fill(par2[3]^2, 1, 1)
Λ̃ = lyap(B̃, Σ̃)
Q̃ = Λ̃ - Φ̃*Λ̃*Φ̃'

gkernel2 = Mitosis.kernel(Gaussian; μ=AffineMap(Φ̃, β̃), Σ=ConstantMap(Q̃))


p = gkernel([u0])

p̂ = Gaussian{(:μ, :Σ)}(mean(samples), cov(samples))
@test norm(p.μ - p̂.μ) < 5/sqrt(K)
@test norm(p.Σ - p̂.Σ) < 5/sqrt(K)



m, p = Mitosis.backwardfilter(gkernel, y)
# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, y_, 0.0]; # start with observation μ=y with uncertainty Σ=0
NT = NamedTuple{mynames}(myvalues)

message, solend = MitosisStochasticDiffEq.backwardfilter(sdekernel, NT)

@testset "Mitosis backward" begin
    @test (p.c)[] ≈ solend[3]
    @test (p.Γ\p.F)[] ≈ solend[1] atol=atol
    @test inv(p.Γ)[] ≈ solend[2] atol=atol
end

# Try forward
V = WGaussian{(:F, :Γ, :c)}(y*5.0, 5.0I, 0.0)
m, p = Mitosis.backwardfilter(gkernel, V)
kᵒ = Mitosis.left′(BFFG(), gkernel, m)

pT = kᵒ([u0])

# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, y_, 0.2]; # start with observation μ=y with uncertainty Σ=0.1
NT = NamedTuple{mynames}(myvalues)
message, solend = MitosisStochasticDiffEq.backwardfilter(sdekernel, NT)

solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (u0, 0.0), Z=nothing; save_noise=true)

@test ll == 0

samples = [MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (u0, 0.0), Z=nothing; save_noise=true)[1][1,end] for k in 1:K]

@testset "Mitosis forward" begin
    @test pT.μ[] ≈ mean(samples) atol=atol
    @test pT.Σ[] ≈ cov(samples) atol=atol

end


# try tilted forward


m, p2 = Mitosis.backwardfilter(gkernel2, V)
gᵒ = Mitosis.left′(BFFG(), gkernel, gkernel2, m, [u0])

message, solend = MitosisStochasticDiffEq.backwardfilter(sdekernel2, NT)
@testset "Mitosis backward tilted" begin
    @test (p2.c)[] ≈ solend[3]
    @test (p2.Γ\p2.F)[] ≈ solend[1] atol=atol
    @test inv(p2.Γ)[] ≈ solend[2] atol=atol
end

solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel2, message, (u0, 0.0), Z=nothing; save_noise=true)


we((x,c)) = x*exp(c)
samples2 = [MitosisStochasticDiffEq.forwardguiding(sdekernel2, message, (u0, 0.0), Z=nothing; save_noise=true)[1][:,end] for k in 1:K]

samples = we.(samples2)



@show std(samples)
ptrue = Mitosis.density(p, [u0])
p̃ = Mitosis.density(WGaussian{(:F,:Γ,:c)}(solend[2]\solend[1], inv(solend[2]), solend[3]), u0)
@testset "Mitosis tilted forward" begin
    @test pT.μ[] ≈ mean(samples)*p̃/ptrue atol=atol
    @test gᵒ.μ[] ≈ mean(first.(samples2)) atol=2atol
    @test gᵒ.Σ[] ≈ cov(first.(samples2)) atol=2atol
    @test gᵒ.c[] ≈ mean(last.(samples2)) atol=atol
end
