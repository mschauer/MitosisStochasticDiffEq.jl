function construct_sample_Problem(k::SDEKernel, u0, Z, alg::Union{StochasticDiffEqAlgorithm,StochasticDiffEqRODEAlgorithm})
  @unpack f, g, trange, p, noise_rate_prototype = k

  prob = SDEProblem(f, g, u0, get_tspan(trange), p, noise=Z,
                    noise_rate_prototype=noise_rate_prototype)

  return prob
end

function construct_sample_Problem(k::SDEKernel, u0, Z, alg::AbstractInternalSolver)
  @unpack f, g, trange, p, noise_rate_prototype = k

  @assert u0 isa AbstractVector

  Z = compute_Z(Z,noise_rate_prototype,trange,u0)

  u = (1, trange[1], deepcopy(u0))
  dz = (0, trange[1], Z[1])
  du = (0, trange[1], zero(u0))

  return u, dz, du, Z
end


function sample(k::SDEKernel, u0, alg=EM(false), Z=nothing; kwargs...)
  return _sample(k, u0, alg, Z; kwargs...)
end

function _sample(k::SDEKernel, u0, alg::Union{StochasticDiffEqAlgorithm,StochasticDiffEqRODEAlgorithm}, Z; kwargs...)
  @unpack f, g, trange, p = k

  prob = construct_sample_Problem(k, u0, Z, alg)
  sol = solve(prob, alg, dt = get_dt(trange); kwargs...)
  return sol, sol[end]
end


function _sample(k::SDEKernel, u0, alg::AbstractInternalSolver, Z; P=k, save=true, kwargs...)
  @unpack f, g, trange, p = k

  u, dz, du, Z = construct_sample_Problem(k, u0, Z, alg)
  if save
    uu = typeof(u)[]
  else
    uu = nothing
  end
  uu, uT = solve!(alg, uu, u, Z, P)
  return uu, uT
end


# Ensemble Problem
function sample(k::SDEKernel, u0, numtraj::Number, alg=EM(false), Z=nothing, ensemblealg=EnsembleThreads(),
    output_func=(sol,i) -> (sol[end],false); kwargs...)
  @unpack f, g, trange, p = k

  prob = construct_sample_Problem(k, u0, Z, alg)
  ensembleprob = EnsembleProblem(prob, output_func = output_func)
  
  sol = solve(ensembleprob, alg, ensemblealg, dt = get_dt(trange), trajectories=numtraj; kwargs...)
  return sol
end


# DefaultSampler for SDEKernel
function tangent!(du, u, dz, P::SDEKernel)
  @unpack f, g, p, noise_rate_prototype = P
  k1 = f(u[3],p,u[2])
  g1 = g(u[3],p,u[2])
  if noise_rate_prototype===nothing
    @. du[3] = k1*dz[2] + g1*dz[3]
  else
    du[3] .= k1*dz[2] + g1*dz[3]
  end
  (dz[1], dz[2], du[3])
end

function exponential_map!(u, du, P::SDEKernel)
  x = u[3]
  @. x += du[3]
  (u[1] + du[1], u[2] + du[2], x)
end
