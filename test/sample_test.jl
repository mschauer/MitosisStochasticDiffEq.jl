using MitosisStochasticDiffEq
using Test

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
kernel = MitosisStochasticDiffEq.SDEKernel(f,g,u0,tstart,tend,pest,plin,p=p,dt=dt)

# sample using MitosisStochasticDiffEq and EM default
sol = MitosisStochasticDiffEq.sample(kernel, save_noise=true)


"""
forwardsample(f, g, p, s, W, x) using the Euler-Maruyama scheme
on a time-grid s with associated noise values W
"""
function forwardsample(f, g, p, s, Ws, x)
    xs = typeof(x)[]
    for i in eachindex(s)[1:end-1]
        dt = s[i+1] - s[i]
        push!(xs, x)
        x = x + f(x, p, s[i])*dt + g(x, p, s[i])*(Ws[i+1]-Ws[i])
    end
    push!(xs, x)

    return xs
end

@test isapprox(sol.u, forwardsample(f,g,p,sol.t,sol.W.W,sol.prob.u0), atol=1e-12)
