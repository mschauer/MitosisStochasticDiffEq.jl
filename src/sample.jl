function sample(k::SDEKernel, u0; alg=EM(false),kwargs...)
    @unpack f, g, tstart, tend, pest, dt = k
    prob = SDEProblem(f, g, u0, (tstart,tend), pest)
    sol = solve(prob, alg, dt = dt; kwargs...)
    return sol, sol[end]
end

function sample(k::SDEKernel, u0, numtraj, ensemblealg=EnsembleThreads(), output_func=(sol,i) -> (sol[end],false); alg=EM(false),kwargs...)
    @unpack f, g, tstart, tend, pest, dt = k
    prob = SDEProblem(f, g, u0, (tstart,tend), pest)
    ensembleprob = EnsembleProblem(prob, output_func = output_func)

    sol = solve(ensembleprob, alg, ensemblealg, dt = dt, trajectories=numtraj; kwargs...)
    return sol
end
