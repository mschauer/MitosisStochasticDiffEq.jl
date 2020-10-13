module MitosisStochasticDiffEq

using Mitosis
using StochasticDiffEq
using OrdinaryDiffEq
using DiffEqCallbacks
using DiffEqNoiseProcess
using LinearAlgebra
using Random
using Parameters
using Statistics

struct SDEKernel{T}
    params::T
end


function Mitosis.sample(k::SDEKernel, u0; method=EM(false))
    @unpack f, g, u0, trange, p, dt = k.params
    prob = SDEProblem(f, g, u0, trange, p)
    sol = solve(prob, method, dt = dt)
    return sol
end


function backwardfilter(k::SDEKernel, (c, ν, P)::NamedTuple{(:logscale, :μ, :Σ)})
    @unpack trange, p, dt = k.params

    # Initialize OD
    u0 = [ν, P, c]

    function filterODE(u, p, t)
      B, β, σtil = p

      # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
      ν, P, c = u

      H = inv(P)
      F = H*ν

      dP = B*P + P*B' - σtil*σtil'
      dν = B*ν + β
      dc = tr(B)

      return [dν, dP, dc]
    end

    prob = ODEProblem(filterODE, u0, reverse(trange), p)
    sol = solve(prob, Euler(), dt=dt)
    message = sol
    return sol[end], message
end


function forwardguiding(k::SDEKernel, message, (u0, ll), Z=WienerProcess(0.0,[0.0],nothing))
    @unpack f, g, trange, p, dt = k.params
    guided_f = let sol=message, cur_time=cur_time
      function ((u,ll), p,t)

        # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
        # ν, P, c
        νPc = @view sol[cur_time[]][:]
        cur_time[] += 1
        P = νPc[2]
        ν = νPc[1]
        r = inv(P)*(ν .- u)
        dll = dot(b(x,pest,si) - b̃(x,ptilde,si), r) - 0.5*tr((σ(x,pest,si)*σ(x,pest,si)' - σ̃(x,ptilde,si)*σ̃(x,ptilde,si)')*(inv(P) .- r*r'))*dt    
        du = f(u, p, t) + g(u, p, t)*g(u, p, t)'*r  # evolution guided by observations
        return [du, dll]
      end
    end

    prob = SDEProblem(guided_f, g, (u0, ll), trange, noise=Z)
    sol = solve(prob, EM(false), dt=dt, adaptive=false)
    return vec(Array(sol(s)))
end


end
