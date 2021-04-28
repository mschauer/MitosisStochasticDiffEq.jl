function construct_sample_Problem(k::SDEKernel, u0, Z)
  @unpack f, g, trange, p, noise_rate_prototype = k

  prob = SDEProblem(f, g, u0, get_tspan(trange), p, noise=Z,
                    noise_rate_prototype=noise_rate_prototype)

  return prob
end

function sample(k::SDEKernel, u0; Z=nothing, alg=EM(false),kwargs...)
  @unpack f, g, trange, p = k

  prob = construct_sample_Problem(k, u0, Z)
  sol = solve(prob, alg, dt = get_dt(trange); kwargs...)
  return sol, sol[end]
end

function sample(k::SDEKernel, u0, numtraj, ensemblealg=EnsembleThreads(),
    output_func=(sol,i) -> (sol[end],false); Z=nothing, alg=EM(false),kwargs...)
  @unpack f, g, trange, p = k

  prob = construct_sample_Problem(k, u0, Z)
  ensembleprob = EnsembleProblem(prob, output_func = output_func)

  sol = solve(ensembleprob, alg, ensemblealg, dt = get_dt(trange), trajectories=numtraj; kwargs...)
  return sol
end
