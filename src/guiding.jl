function range2ind(ts::AbstractRange, t)
  indx = round(Int, (t-first(ts))/step(ts))
  indx += one(indx)
  indx = minimum((maximum((indx,one(indx))),length(ts)))
  return indx
end

function range2ind(ts::AbstractVector, t)
  r = searchsorted(ts, t)
  r1 = minimum((first(r),length(ts)))
  r2 = maximum((last(r),one(last(r))))
  return abs(ts[r1] - t) < abs(ts[r2] - t) ? r1 : r2
end

unpackx(a) = @view a[1:end-1]
mypack(a::SArray,c::Number) = SVector(a..., c)
mypack(a,c::Number) = [a; c]

# guided drift
function (G::GuidingDriftCache)(du,u,p,t)
  @unpack k, message = G
  @unpack f, g, constant_diffusity = k
  @unpack ktilde, ts, soldis, sol, filter = message

  x = unpackx(u)
  dx = unpackx(du)
  d = length(x)

  # find cursor
  @inbounds cur_time = range2ind(ts, t)

  if isapprox(t, ts[cur_time]; atol = 1000eps(typeof(t)), rtol = 1000eps(t))
    # non-interpolating version
    # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
    # ν, P, c or F, H, c parametrization = μ,Σ,c
    μ = @view soldis[1:d,cur_time]
    Σ = reshape(@view(soldis[d+1:d+d*d,cur_time]), d, d)
  else
    μ = @view sol(t)[1:d]
    Σ = reshape(@view(sol(t)[d+1:d+d*d]), d, d)
  end

  if !(filter isa InformationFilter)
    r = Σ\(μ - x)
    du[end] = dot(f(x,p,t) - ktilde.f(x,ktilde.p,t), r)
    if !constant_diffusity
      du[end] -= 0.5*tr((outer_(g(x,p,t)) - outer_(ktilde.g(x,ktilde.p,t)))*(inv(Σ) - outer_(r)))
    end
    dx[:] .= vec(f(x,p,t) + (outer_(g(x,p,t))*r)) # evolution guided by observations
  else
    du[end] = ..
    dx[:] .= ..
  end

  return nothing
end

function (G::GuidingDriftCache)(u,p,t)
  @unpack k, message = G
  @unpack f, g, constant_diffusity = k
  @unpack ktilde, ts, soldis, sol = message

  x = unpackx(u)
  d = length(x)

  # find cursor
  @inbounds cur_time = range2ind(ts, t)

  if isapprox(t, ts[cur_time]; atol = 1000eps(typeof(t)), rtol = 1000eps(t))
    # non-interpolating version
    # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
    # ν, P, c or F, H, c parametrization = μ,Σ,c
    μ = @view soldis[1:d,cur_time]
    Σ = reshape(@view(soldis[d+1:d+d*d,cur_time]), d, d)
  else
    μ = @view sol(t)[1:d]
    Σ = reshape(@view(sol(t)[d+1:d+d*d]), d, d)
  end

  if !(filter isa InformationFilter)
    r = Σ\(μ .- x)

    dl = dot(f(x,p,t) -  ktilde.f(x,ktilde.p,t), r)
    if !constant_diffusity
      dl -= 0.5*tr((outer_(g(x,p,t)) - outer_(ktilde.g(x,ktilde.p,t)))*(inv(Σ) - outer_(r)))
    end
    dx = vec(f(x,p,t) + outer_(g(x,p,t))*r) # evolution guided by observations
  else
    dl = ..
    dx = ..
  end

  return mypack(dx, dl)
end

# guided diffusion
function (G::GuidingDiffusionVectorCache)(du,u,p,t)
  @unpack g = G

  x = @view u[1:end-1]
  du[1:end-1] .= g(x,p,t)
  return nothing
end

function (G::GuidingDiffusionVectorCache)(u,p,t)
  @unpack g = G

  x = @view u[1:end-1]
  dx = g(x,p,t)
  return mypack(dx, zero(eltype(u)))
end

function (G::GuidingDiffusionCache)(du,u,p,t)
  @unpack g = G

  x = @view u[1:end-1]
  du[1:end-1,1:end-1] .= g(x,p,t)
  return nothing
end


function (G::GuidingDiffusionCache)(u,p,t)
  @unpack g, padded_size = G

  x = @view u[1:end-1]
  dx = g(x,p,t)
  return PaddedView(zero(eltype(dx)), dx, padded_size)
end

function construct_forwardguiding_Problem(k::SDEKernel, message, u0, Z, inplace)
  @unpack f, g, trange, p, noise_rate_prototype = k

  guided_f = GuidingDriftCache(k,message)

  if noise_rate_prototype===nothing
    # diagonal noise in StochasticDiffEq
    guided_g = GuidingDiffusionVectorCache(g)
    _noise_rate_prototype = nothing
  else
    # non-diagonal noise in StochasticDiffEq
    padded_size = size(noise_rate_prototype) .+ 1
    _noise_rate_prototype = PaddedView(zero(eltype(noise_rate_prototype)), noise_rate_prototype, padded_size)
    guided_g = GuidingDiffusionCache(g,padded_size)
  end

  prob = SDEProblem{inplace}(guided_f, guided_g, u0, get_tspan(trange), p, noise=Z,
                             noise_rate_prototype=_noise_rate_prototype)

  return prob
end


function forwardguiding(k::SDEKernel, message, (x0, ll0), Z=nothing; alg=EM(false),
    dt=get_dt(k.trange), isadaptive=StochasticDiffEq.isadaptive(alg),
    numtraj=nothing, ensemblealg=EnsembleThreads(), output_func=(sol,i) -> (sol,false),
    inplace=true, kwargs...)

  @unpack f, g, trange, p = k

  u0 = mypack(x0,ll0)

  # check that message.ts is sorted
  !issorted(message.ts) && error("Something went wrong. Message.ts is not sorted! Please report this.")

  # construct guiding SDE problem
  prob = construct_forwardguiding_Problem(k::SDEKernel, message, u0, Z, inplace)

  if numtraj==nothing
    if !isadaptive
      sol = solve(prob, alg, tstops=message.ts; kwargs...)
    else
      sol = solve(prob, alg, dt=dt, adaptive=isadaptive; kwargs...)
    end
  else
    ensembleprob = EnsembleProblem(prob, output_func = output_func)
    if !isadaptive
      sol = solve(ensembleprob, alg, ensemblealg=ensemblealg,
        tstops=message.ts, trajectories=numtraj; kwargs...)
    else
      sol = solve(ensembleprob, alg, ensemblealg=ensemblealg,
        dt=dt, adaptive=isadaptive, trajectories=numtraj; kwargs...)
    end
  end

  return sol, sol[end][end]
end
