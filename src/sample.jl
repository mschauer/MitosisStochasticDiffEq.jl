function sample(k::SDEKernel, u0; alg=EM(false),kwargs...)
  @unpack f, g, trange, p = k
  prob = SDEProblem(f, g, u0, get_tspan(trange), p)
  sol = solve(prob, alg, dt = get_dt(trange); kwargs...)
  return sol, sol[end]
end

function sample(k::SDEKernel, u0, numtraj, ensemblealg=EnsembleThreads(),
    output_func=(sol,i) -> (sol[end],false); alg=EM(false),kwargs...)
  @unpack f, g, trange, p = k
  prob = SDEProblem(f, g, u0, get_tspan(trange), p)
  ensembleprob = EnsembleProblem(prob, output_func = output_func)

  sol = solve(ensembleprob, alg, ensemblealg, dt = get_dt(trange), trajectories=numtraj; kwargs...)
  return sol
end
