# linear approximation
function b̃(u,p,t)
    p[1]*u .+ p[2]
end

function σ̃(u,p,t)
    p[3]
end

function forwardguiding(k::SDEKernel, message, (x0, ll0), Z=nothing; alg=EM(false),
  numtraj=nothing, ensemblealg=EnsembleThreads(), output_func=(sol,i) -> (sol[end],false), kwargs...)
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

        du[end] = dot(f(x,p,ti) -  b̃(x,ptilde,ti), r) - 0.5*tr((g(x,p,ti)*g(x,p,ti)' .- σ̃(x,ptilde,ti)*σ̃(x,ptilde,ti)')*(inv(P) .- r*r'))
        dx[:] .= vec(f(x, p, ti) .+ g(x, p, ti)*g(x, p, ti)'*r) # evolution guided by observations
        return nothing
      end
    end

    function guided_g(du,u,p,t)
      x = @view u[1:end-1]

      du[1:end-1] .= g(x,p,t)
      return nothing
    end

    if Z!=nothing
      prob = SDEProblem(guided_f, guided_g, u0, trange, pest, noise=Z)
    else
      prob = SDEProblem(guided_f, guided_g, u0, trange, pest)
    end

    if numtraj==nothing
      sol = solve(prob, alg, dt=dt; kwargs...)
    else
      ensembleprob = EnsembleProblem(prob, output_func = output_func, prob_func=()->cur_time[]=1)
      sol = solve(ensembleprob, alg, ensemblealg, dt = dt, trajectories=numtraj; kwargs...)
    end
    return sol, sol[end][end]
end
