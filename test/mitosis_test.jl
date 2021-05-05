import MitosisStochasticDiffEq as MSDE
using Mitosis, Statistics
using LinearAlgebra, Test
using Random
Random.seed!(126)
using Mitosis: AffineMap
atol = 0.015
# set true model parameters
par = [-0.2, 0.1, 0.9]

# Samples
K = 5000

# define SDE function
f(u,p,t) = p[1]*u .+ p[2]
g(u,p,t) = p[3] .+ 0u # needs to preserve type of u for forwardguided

# time span
tstart = 0.0
tend = 1.0
dt = 0.01
trange = tstart:dt:tend

# initial condition
u0 = 1.1

# set of linear parameters Eq.~(2.2)
plin = copy(par)
sdekernel = MSDE.SDEKernel(f,g,trange,plin)

plin2 = par2 = [-0.1, -0.05, 1.0]
sdekernel2 = MSDE.SDEKernel(f,g,trange,plin)
sdekerneltilde2 = MSDE.SDEKernel(f,g,trange,plin2)



sol, y_ = MSDE.sample(sdekernel, u0, save_noise=true)

y_ = 1.39
y = vcat(y_)
samples_ = MSDE.sample(sdekernel, u0, K, save_noise=true).u

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



m, l = Mitosis.backwardfilter(gkernel, y) # returns a "Leaf"
p = l.y
# initial values for ODE
WG = WGaussian{(:μ,:Σ,:c)}(y_, 0.0, 0.0) #

message, solend = MSDE.backwardfilter(sdekernel, WG)
@testset "Mitosis backward" begin
    @test (p.c)[] ≈ solend.c
    @test (p.Γ\p.F)[] ≈ solend.μ atol=atol
    @test inv(p.Γ)[] ≈ solend.Σ atol=atol
end

# Try forward
V = WGaussian{(:F, :Γ, :c)}(y*5.0, 5.0I, 0.0)
m, p = Mitosis.backwardfilter(gkernel, V)
kᵒ = Mitosis.left′(BFFG(), gkernel, m)

pT = kᵒ([u0])

# initial values for ODE
WG = WGaussian{(:μ,:Σ,:c)}(y_, 0.2, 0.0) # start with observation μ=y with uncertainty Σ=0.1
message, solend = MSDE.backwardfilter(sdekernel, WG)

solfw, ll = MSDE.forwardguiding(sdekernel, message, (u0, 0.0), Z=nothing; save_noise=true)

@test ll == 0

samples = [MSDE.forwardguiding(sdekernel, message, (u0, 0.0), Z=nothing; save_noise=true)[1][1,end] for k in 1:K]

@testset "Mitosis forward" begin
    @test pT.μ[] ≈ mean(samples) atol=atol
    @test pT.Σ[] ≈ cov(samples) atol=atol
end


# try tilted forward

m, p2 = Mitosis.backwardfilter(gkernel2, V)
gᵒ = Mitosis.forward(BFFG(), gkernel, m, [u0])

message, solend = MSDE.backwardfilter(sdekerneltilde2, WG)
@testset "Mitosis backward tilted" begin
    @test (p2.c)[] ≈ solend.c
    @test (p2.Γ\p2.F)[] ≈ solend.μ atol=atol
    @test inv(p2.Γ)[] ≈ solend.Σ atol=atol
end

solfw, ll = MSDE.forwardguiding(sdekernel2, message, (u0, 0.0), Z=nothing; save_noise=true)

we((x,c)) = x*exp(c)
samples2 = [MSDE.forwardguiding(sdekernel2, message, (u0, 0.0), Z=nothing; save_noise=true)[1][:,end] for k in 1:K]

samples = we.(samples2)

@show std(samples)
ptrue = Mitosis.density(p, [u0])
p̃ = Mitosis.density(WGaussian{(:F,:Γ,:c)}(solend.Σ\solend.μ, inv(solend.Σ), solend.c), u0)
@testset "Mitosis tilted forward" begin
    @test pT.μ[] ≈ mean(samples)*p̃/ptrue atol=atol
    @test gᵒ.μ[] ≈ mean(first.(samples2)) atol=2atol
    @test gᵒ.Σ[] ≈ cov(first.(samples2)) atol=2atol
    @test gᵒ.c[] ≈ mean(last.(samples2)) atol=atol
end
