module MitosisStochasticDiffEq

using Mitosis
using StochasticDiffEq
using OrdinaryDiffEq
using DiffEqCallbacks
using DiffEqNoiseProcess
using LinearAlgebra
using Random
#using Parameters
using UnPack
using Statistics

struct SDEKernel{fType,gType,tType,dtType,paramType1,paramType2}
    f::fType
    g::gType
    tstart::tType
    tend::tType
    dt::dtType
    p::paramType1
    pest::paramType2
end

function SDEKernel(f,g,u0,tstart,tend,pest;p=nothing,dt=nothing)
  SDEKernel{typeof(f),typeof(g),typeof(tstart),
            typeof(dt),typeof(p),typeof(pest)}(f,g,tstart,tend,dt,p,pest)
end

function sample(k::SDEKernel, u0; alg=EM(false),kwargs...)
    @unpack f, g, tstart, tend, p, dt = k
    prob = SDEProblem(f, g, u0, (tstart,tend), p)
    sol = solve(prob, alg, dt = dt; kwargs...)
    return sol, sol[end]
end


function backwardfilter(k::SDEKernel, (c, ν, P)::NamedTuple{(:logscale, :μ, :Σ)}; alg=Euler())
    @unpack tstart, tend, pest, dt = k

    trange = (tend, tstart)

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

    prob = ODEProblem(filterODE, u0, trange, pest)
    sol = solve(prob, alg, dt=dt)
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
        dll = dot(b(x,pest,si) - b̃(x,ptilde,si), r) - 0.5*tr((σ(x,pest,si)*σ(x,pest,si)' - σ̃(x,ptilde,si)*σ̃(x,ptilde,si)')*(inv(P) .- r*r'))
        du = f(u, p, t) + g(u, p, t)*g(u, p, t)'*r  # evolution guided by observations
        return [du, dll]
      end
    end

    prob = SDEProblem(guided_f, g, (u0, ll), trange, noise=Z)
    sol = solve(prob, EM(false), dt=dt, adaptive=false)
    return sol, sol[end]
end


end
