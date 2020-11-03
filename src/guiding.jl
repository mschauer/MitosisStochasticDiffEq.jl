# linear approximation
function b̃(u,p,t)
  p[1]*u .+ p[2]
end

function σ̃(u,p,t)
  p[3]
end

function forwardguiding(k::SDEKernel, message, (x0, ll0), Z=nothing; alg=EM(false),
  numtraj=nothing, ensemblealg=EnsembleThreads(), output_func=(sol,i) -> (sol,false), kwargs...)
    @unpack f, g, tstart, tend, pest, plin, dt = k

    trange = (tstart, tend)
    u0 = [x0; ll0]

    # non-interpolating version
    guided_f = let sol=reverse(Array(message), dims=2), ts = reverse(message.t), ptilde=plin
      function (du,u,p,t)

        x = @view u[1:end-1]
        dx =  @view du[1:end-1]

        # find cursor
        @inbounds cur_time = searchsortedfirst(ts,t-10eps(typeof(t)),rev=false)

        if isapprox(t, ts[cur_time]; atol = 100eps(typeof(t)), rtol = 100eps(t))
          # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
          # ν, P, c
          ν, P, _ = sol[:,cur_time]
          r = inv(P)*(ν .- x)

          du[end] = dot(f(x,p,t) -  b̃(x,ptilde,t), r) - 0.5*tr((g(x,p,t)*g(x,p,t)' .- σ̃(x,ptilde,t)*σ̃(x,ptilde,t)')*(inv(P) .- r*r'))
          dx[:] .= vec(f(x, p, t) .+ g(x, p, t)*g(x, p, t)'*r) # evolution guided by observations
        else
          error("Interpolation in forwardguiding is not yet implemented.")
        end

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
      ensembleprob = EnsembleProblem(prob, output_func = output_func)
      sol = solve(ensembleprob, alg, ensemblealg=ensemblealg, dt=dt, trajectories=numtraj)
    end

    return sol, sol[end][end]
end
