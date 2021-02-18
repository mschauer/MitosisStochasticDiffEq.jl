using MitosisStochasticDiffEq
using Test

# set estimate of model parameters or true model parameters
p = [-0.1,0.2,0.9]
# define SDE function
f(u,p,t) = p[1]*u + p[2] - 1.5*sin.(u*2pi)
g(u,p,t) = p[3] .- 0.2*(1 .-sin.(u))

# time range
tstart = 0.0
tend = 1.0
dt = 0.02
trange = tstart:dt:tend

# intial condition
u0 = 1.1


kernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,p)
# sample using MitosisStochasticDiffEq and EM default
sol, solend = MitosisStochasticDiffEq.sample(kernel, u0)

kernel = MitosisStochasticDiffEq.SDEKernel(f,g,collect(trange),p)
sol, solend = MitosisStochasticDiffEq.sample(kernel, u0, save_noise=true)


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
