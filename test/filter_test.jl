using MitosisStochasticDiffEq
using Test
using LinearAlgebra

# set true model parameters
p = [-0.1,0.2,0.9]
# define SDE function
f(u,p,t) = p[1]*u + p[2] - 1.5*sin.(u*2pi)
g(u,p,t) = p[3] .- 0.2*(1 .-sin.(u))

# time span
tstart = 0.0
tend = 1.0
dt = 0.02

# intial condition
u0 = 1.1

# set of linear parameters Eq.~(2.2)
plin = [-0.1,0.2,1.3]
pest = [2.0] # initial guess of parameter to be estimated

kernel = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin,p=p,dt=dt)


# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, 0.0, 10.0];
NT = NamedTuple{mynames}(myvalues)

solend, message = MitosisStochasticDiffEq.backwardfilter(kernel, NT)



"""
    backwardfilter() -> ps, p0, c
Backward filtering using the Euler method, starting with `N(ν, P)` prior
and integration scale c2 between observations

"""
function backwardfilter((c, ν, P)::NamedTuple{(:logscale, :μ, :Σ)}, p, s)
    ps = [[ν, P, c]]
    B, β, σ̃ = p
    for i in eachindex(s)[end-1:-1:1]
        dt = s[i+1] - s[i]
        H = inv(P)
        F = H*ν
        P = P - dt*(B*P + P*B' - σ̃*σ̃')
        ν = ν - dt*(B*ν + β)
        c = c - dt*tr(B)
        push!(ps, [ν, P, c])
    end
    [ν, P, c], ps
end


solend2, message2 = backwardfilter(NT, plin, reverse(message.t))

@test isapprox(solend, solend2, rtol=1e-12)
@test isapprox(Array(message), reduce(hcat, message2), rtol=1e-12)
