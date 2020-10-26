module MitosisStochasticDiffEq

using Mitosis
using RecursiveArrayTools
using StochasticDiffEq
using OrdinaryDiffEq
using DiffEqCallbacks
using DiffEqNoiseProcess
using LinearAlgebra
using Random
#using Parameters
using UnPack
using Statistics

struct SDEKernel{fType,gType,tType,dtType,paramType1,paramType2,paramType3}
    f::fType
    g::gType
    tstart::tType
    tend::tType
    dt::dtType
    p::paramType1
    pest::paramType2
    plin::paramType3
end

function SDEKernel(f,g,tstart,tend,pest,plin;p=nothing,dt=nothing)
  SDEKernel{typeof(f),typeof(g),typeof(tstart),
            typeof(dt),typeof(p),typeof(pest),typeof(plin)}(f,g,tstart,tend,dt,p,pest,plin)
end

function sample(k::SDEKernel, u0; alg=EM(false),kwargs...)
    @unpack f, g, tstart, tend, p, dt = k
    prob = SDEProblem(f, g, u0, (tstart,tend), p)
    sol = solve(prob, alg, dt = dt; kwargs...)
    return sol, sol[end]
end

myunpack(a) = a
myunpack(a::ArrayPartition) = a.x
mypack(a,b,c) = ArrayPartition(a,b,[c])
mypack(a::Number...) = [a...]

function filterODE(u, p, t)
  B, β, σtil = p

  # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
  ν, P, c = myunpack(u)

  H = inv(P)
  F = H*ν

  dP = B*P + P*B' .- σtil*σtil'
  dν = B*ν .+ β
  dc = tr(B)

  return mypack(dν, dP, dc)
end

function filterODE(du, u, p, t)
  B, β, σtil = p

  # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
  ν, P, c = myunpack(u)

  H = inv(P)
  F = H*ν

  du.x[1] .= B*ν .+ β
  du.x[2] .= B*P + P*B' .- σtil*σtil'
  du.x[3] .= tr(B)

  return nothing
end

function backwardfilter(k::SDEKernel, (c, ν, P)::NamedTuple{(:logscale, :μ, :Σ)}; alg=Euler(), inplace=false)
    @unpack tstart, tend, plin, dt = k

    trange = (tend, tstart)

    # Initialize OD
    u0 = mypack(ν, P, c)

    prob = ODEProblem{inplace}(filterODE, u0, trange, plin)
    sol = solve(prob, alg, dt=dt)
    message = sol
    return sol[end], message
end

# linear approximation
function b̃(u,p,t)
  @. p[1]*u + p[2]
end

function σ̃(u,p,t)
  fill(p[3], size(u))
end

# function b̃(du,u,p,t)
#   @inbounds begin
#     @. du = p[1]*u + p[2]
#   end
#   return nothing
# end
#
# function σ̃(du,u,p,t)
#   @inbounds begin
#     du .= p[3]
#   end
#   return nothing
# end

function forwardguiding(k::SDEKernel, message, (x0, ll0), Z=nothing; alg=EM(false), kwargs...)
    @unpack f, g, tstart, tend, pest, plin, dt = k

    trange = (tstart, tend)
    u0 = [x0; ll0]

    # non-interpolating version
    cur_time = Ref(1)
    guided_f = let sol=reverse(Array(message), dims=2), ts = reverse(message.t), cur_time=cur_time, ptilde=plin
      function (du,u,p,t)

        x = @view u[1:end-1]
        dx =  @view du[1:end-1]
        ll = u[end]

        # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
        # ν, P, c
        ν, P, _ = sol[:,cur_time[]]
        ti = ts[cur_time[]]
        cur_time[] += 1
        r = inv(P)*(ν .- x)

        du[end] = dot(f(x,p,ti) -  b̃(x,ptilde,ti), r) - 0.5*tr((g(x,p,ti)*g(x,p,ti)' - σ̃(x,ptilde,ti)*σ̃(x,ptilde,ti)')*(inv(P) .- r*r'))
        dx[:] .= vec(f(x, p, ti) .+ g(x, p, ti)*g(x, p, ti)'.*r) # evolution guided by observations
        return nothing
      end
    end

    function guided_g(du,u,p,t)
      x = @view u[1:end-1]

      du[1:end-1] .= g(x,p,t)
      du[end] = false*u[end]
      return nothing
    end

    if Z!=nothing
      prob = SDEProblem(guided_f, guided_g, u0, trange, pest, noise=Z)
    else
      prob = SDEProblem(guided_f, guided_g, u0, trange, pest)
    end

    sol = solve(prob, alg, dt=dt; kwargs...)
    return sol, sol[end][end]
end


end
