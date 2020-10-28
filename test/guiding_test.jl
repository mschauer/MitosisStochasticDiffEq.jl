using MitosisStochasticDiffEq
using DiffEqNoiseProcess
using Test, Random
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
dt = 0.001

# intial condition
u0 = 1.1


kernel = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin,p=p,dt=dt)


# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, 0.0, 10.0];
NT = NamedTuple{mynames}(myvalues)

backward, message = MitosisStochasticDiffEq.backwardfilter(kernel, NT)

x0 = randn()
ll0 = randn()

solfw, ll = MitosisStochasticDiffEq.forwardguiding(kernel, message, (x0, ll0),
            Z=nothing; save_noise=true)


"""
    forwardguiding(M, s, x, ps, Z) -> xs, ll
Forward sample a guided trajectory `xs` starting in `x` and compute it's
log-likelihood `ll` with innovations `Z = randn(length(s))`.
"""
function forwardguiding(plin, pest, s, (x, ll), ps, Z=randn(length(s)), noisetype=:scalar)
    # linear approximation of b and constant approximation of σ
    # with parameters B, β, and σ̃
    flinear(u,p,t) = p[1]*u .+ p[2]
    σlinear(u,p,t) = p[3]

    llstep(x, r, t, P) = dot(f(x,pest,t) - flinear(x,plin,t), r) -0.5*tr((g(x,pest,t)*g(x,pest,t)' - σlinear(x,plin,t)*σlinear(x,plin,t)')*(inv(P) .- r*r'))
    xs = typeof(x)[]
    for i in eachindex(s)[1:end-1]
        dt = s[i+1] - s[i]
        t = s[i]
        push!(xs, x)

        ν, P, _ = ps[:,i]
        r = inv(P)*(ν .- x)

        ll += llstep(x, r, t, P)*dt # accumulate log-likelihood
        if noisetype == :scalar
            noise = g(x,pest,t)*Z[i] #sqrt(dt)*Z[i]
        elseif noisetype ==:diag
            noise = g(x,pest,t).*Z[:,i]
        elseif noisetype ==:nondiag
            noise = g(x,pest,t)*Z[:,i]
        else
            error("noisetype not understood.")
        end
        x = x + f(x,pest,t)*dt + g(x,pest,t)*g(x,pest,t)'*r*dt + noise # evolution guided by observations
    end
    push!(xs, x)
    xs, ll
end


dWs = (solfw.W[1,2:end]-solfw.W[1,1:end-1])
ps = reverse(Array(message), dims=2)
solfw2, ll2 = forwardguiding(plin, pest, reverse(message.t), (x0, ll0),ps,dWs)

@test isapprox(solfw[1,:], solfw2, rtol=1e-12)
@test isapprox(ll, ll2, rtol=1e-12)



# multivariate tests with scalar random process
dim = 7
Random.seed!(1234)
logscale = randn()
μ = randn(dim)
Σ = randn(dim,dim)
myvalues = [logscale, μ, Σ];
NT = NamedTuple{mynames}(myvalues)

m = 1
plin = [randn(dim,dim), randn(dim), randn(dim,m)] # B, β, σtil

# set scalar random process
t = tstart:dt:tend
W = sqrt(dt)*randn(length(t))
W1 = cumsum([zero(dt); W[1:end-1]])
NG = NoiseGrid(t,W1)

kernel = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin,p=p,dt=dt)
solend, message = MitosisStochasticDiffEq.backwardfilter(kernel, NT)

x0 = randn(dim)
ll0 = randn()

solfw, ll = MitosisStochasticDiffEq.forwardguiding(kernel, message, (x0, ll0), NG)


ps = reverse(Array(message), dims=2)
solfw2, ll2 = forwardguiding(plin, pest, reverse(message.t), (x0, ll0),ps,W)

@test isapprox(Array(solfw)[1:dim,:], hcat(solfw2 ...), rtol=1e-12)
@test isapprox(ll, ll2, rtol=1e-12)


# multivariate tests with diagonal noise random process

dim = 2
Random.seed!(12345)
logscale = randn()
μ = randn(dim)
Σ = randn(dim,dim)
myvalues = [logscale, μ, Σ];
NT = NamedTuple{mynames}(myvalues)

m = 2
plin = [randn(dim,dim), randn(dim), randn(dim,m)] # B, β, σtil

kernel = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin,p=p,dt=dt)
solend, message = MitosisStochasticDiffEq.backwardfilter(kernel, NT)

x0 = randn(dim)
ll0 = randn()

solfw, ll = MitosisStochasticDiffEq.forwardguiding(kernel, message, (x0, ll0); save_noise=true)

Ws = Array(solfw.W)
dWs = Ws[1:dim,2:end]-Ws[1:dim,1:end-1]

ps = reverse(Array(message), dims=2)
solfw2, ll2 = forwardguiding(plin, pest, reverse(message.t), (x0, ll0),ps,dWs,:diag)

@test isapprox(Array(solfw)[1:dim,:], hcat(solfw2 ...), rtol=1e-12)
@test isapprox(ll, ll2, rtol=1e-12)
