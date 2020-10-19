using MitosisStochasticDiffEq
using Test
using LinearAlgebra

# set true model parameters
p = [-0.1,0.2,0.9]
# define SDE function
f(u,p,t) = @. p[1]*u + p[2] - 1.5*sin(u*2pi)
g(u,p,t) = p[3] .- 0.2*(1 .-sin.(u))

# set of linear parameters Eq.~(2.2)
plin = [-0.1,0.2,1.3]
pest = [-0.4, 0.5, 1.4] # initial guess of parameter to be estimated

# time span
tstart = 0.0
tend = 1.0
dt = 0.02

# intial condition
u0 = 1.1


kernel = MitosisStochasticDiffEq.SDEKernel(f,g,u0,tstart,tend,pest,plin,p=p,dt=dt)


# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, 0.0, 10.0];
NT = NamedTuple{mynames}(myvalues)

backward, message = MitosisStochasticDiffEq.backwardfilter(kernel, NT)

x0 = randn()
ll0 = randn()

MitosisStochasticDiffEq.forwardguiding(kernel, message, (x0, ll0), Z=nothing)



"""
    forwardguiding(M, s, x, ps, Z) -> xs, ll
Forward sample a guided trajectory `xs` starting in `x` and compute it's
log-likelihood `ll` with innovations `Z = randn(length(s))`.
"""
function forwardguiding(plin, pest, s, (x, ll), ps, Z=randn(length(s)))
    # linear approximation of b and constant approximation of σ
    # with parameters B, β, and σ̃
    flinear(u,p,t) = p[1]*x + p[2]
    σlinear(u,p,t) = p[3]

    llstep(x, r, t, P) = dot(f(x,pest,t) - flinear(x,plin,t), r) -0.5*tr((g(x,pest,t)*g(x,pest,t)' -σlinear(x,plin,t)*σlinear(x,plin,t)')*(inv(P) - r*r'))
    xs = typeof(x)[]
    for i in skiplast(eachindex(s))
        dt = s[i+1] - s[i]
        t = s[i]
        push!(xs, x)
        ν, P, _ = ps[i]
        r = inv(P)*(ν - x)
        ll += llstep(x, r, dt, P)*dt # accumulate log-likelihood
        x = x + f(x,pest,t)*dt + g(x,pest,t)*g(x,pest,t)'*r*dt + g(x,pest,t)*sqrt(dt)*Z[i] # evolution guided by observations
    end
    push!(xs, x)
    xs, ll
end

sol2, ll2 = forwardguiding(NT, pest, reverse(message.t))

@test isapprox(solend, solend2, rtol=1e-12)
@test isapprox(Array(message), reduce(hcat, message2), rtol=1e-12)
