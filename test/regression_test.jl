using MitosisStochasticDiffEq
using Mitosis
using Test, Random
using Statistics
using LinearAlgebra


"""
    conjugate_posterior(Y, Ξ)

Sample the posterior distribution of the conjugate drift parameters from path `Y`,
prior precision matrix `Ξ` with non-conjugate parameters fixed in the model.
Adjusted from http://www.math.chalmers.se/~smoritz/journal/2018/01/19/parameter-inference-for-a-simple-sir-model/
for Bridge.jl
"""
function conjugate_posterior(Y, Ξ)
    paramgrad(t, u) = [u, 1]
    paramintercept(t, u) = 0
    t, y = Y.t[1], Y.u[1]
    ϕ = paramgrad(t, y)
    mu = zero(ϕ)
    G = zero(mu*mu')

    for i in 1:length(Y)-1
        ϕ = paramgrad(t, y)'
        Gϕ = pinv(Y.prob.g(y, Y.prob.p, t)*Y.prob.g(y, Y.prob.p, t)')*ϕ # a is sigma*sigma'. Todo: smoothing like this is very slow
        zi = ϕ'*Gϕ
        t2, y2 = Y.t[i + 1], Y.u[i + 1]
        dy = y2 - y
        ds = t2 - t
        #@show size(mu), size(Gϕ'), (dy - paramintercept(t, y)*ds)
        mu = mu + Gϕ'*(dy - paramintercept(t, y)*ds)
        t, y = t2, y2
        G = G +  zi*ds
    end
    Mitosis.Gaussian{(:F,:Γ)}(mu, G + Ξ)
end



K = 1000
Random.seed!(100)

# define SDE function
f(u,p,t) = p[1]*u .+ p[2]
g(u,p,t) = p[3]

# paramjac
function f_jac(J,u,p,t)
  J[1,1] = u[1]
  J[1,2] = true
  nothing
end
ϕprototype = zeros((1,2))
yprototype = zeros((1))

# intercept
function ϕ0(du,u,p,t)
  du .= false
end

# time span
tstart = 0.0
tend = 200.0
dt = 0.01

# initial condition
u0 = 1.1

# set true model parameters
par = [-0.3, 0.2, 0.5]

# set of linear parameters Eq.~(2.2)
plin = copy(par)
pest = copy(par)
sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin,dt=dt)

# sample using MitosisStochasticDiffEq and EM default
sol, solend = MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=true)

R = MitosisStochasticDiffEq.Regression(sdekernel,yprototype,ϕprototype,paramjac=f_jac,intercept=ϕ0)


Π = []
Π2 = []

G = MitosisStochasticDiffEq.conjugate(R, sol, 0.1*I(2))
G2 = conjugate_posterior(sol, 0.1*I(2))

mu = G.F
Gamma = G.Γ
WL = (cholesky(Hermitian(Gamma)).U)'

Random.seed!(1)
for i=1:K
  th° = WL'\(randn(size(mu))+WL\mu)
  push!(Π,th°)
  th° = WL'\(randn(size(mu))+WL\mu)
  push!(Π2,th°)
end

mu = G2.F
Gamma = G2.Γ
WL = (cholesky(Hermitian(Gamma)).U)'

Random.seed!(1)
for i=1:K
  th° = WL'\(randn(size(mu))+WL\mu)
  push!(Π2,th°)
end




@test par[1:2] ≈ mean(Π) rtol=0.2
@test par[1:2] ≈ mean(Π2) rtol=0.2
@test mean(Π) ≈ mean(Π2) atol=0.1

# using Plots
# pl = scatter(first.(Π), last.(Π), markersize=1, c=:blue, label="posterior samples")
# scatter!(first.(Π2), last.(Π2), markersize=1, c=:green, label="posterior samples")
# scatter!([par[1]], [par[2]], color="red", label="truth")
# savefig(pl, "regression.png")
