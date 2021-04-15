function sample(k::SDEKernel, u0; Z=nothing, alg=EM(false),kwargs...)
  @unpack f, g, trange, p = k
  if Z===nothing
    prob = SDEProblem(f, g, u0, get_tspan(trange), p)
  else
    prob = SDEProblem(f, g, u0, get_tspan(trange), p, noise=Z)
  end
  sol = solve(prob, alg, dt = get_dt(trange); kwargs...)
  return sol, sol[end]
end

function sample(k::SDEKernel, u0, numtraj, ensemblealg=EnsembleThreads(),
    output_func=(sol,i) -> (sol[end],false); Z=nothing, alg=EM(false),kwargs...)
  @unpack f, g, trange, p = k

  if Z===nothing
    prob = SDEProblem(f, g, u0, get_tspan(trange), p)
  else
    prob = SDEProblem(f, g, u0, get_tspan(trange), p, noise=Z)
  end

  ensembleprob = EnsembleProblem(prob, output_func = output_func)

  sol = solve(ensembleprob, alg, ensemblealg, dt = get_dt(trange), trajectories=numtraj; kwargs...)
  return sol
end
