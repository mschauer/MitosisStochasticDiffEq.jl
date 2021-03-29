myunpack(a) = a
myunpack(a::ArrayPartition) = (a.x[1], a.x[2], a.x[3][])
mypack(a,b,c) = ArrayPartition(a,b,[c])
mypack(a::Number...) = [a...]
mypack(a::SArray,b::SArray,c::Number) = ArrayPartition(a,b,@SVector[c])
mypack(a::SArray,b::SArray,c::SArray) = ArrayPartition(a,b,c)

function backwardfilter(k::SDEKernel, p::WGaussian{(:μ, :Σ, :c)};
    filter=CovarianceFilter(), alg=Euler(),
    inplace=false, apply_timechange=false, abstol=1e-6, reltol=1e-3)

  message, solend = _backwardfilter(filter, k::SDEKernel, (p.c, p.μ, p.Σ);
    alg=alg, inplace=inplace, apply_timechange=apply_timechange, abstol=abstol, reltol=reltol)

  return message, WGaussian{(:μ, :Σ, :c)}(myunpack(solend)...)
end

function backwardfilter(k::SDEKernel, (c, μ, Σ)::NamedTuple{(:logscale, :μ, :Σ)};
    filter=CovarianceFilter(), alg=Euler(), inplace=false, apply_timechange=false, abstol=1e-6,reltol=1e-3)

  return _backwardfilter(filter, k::SDEKernel, (c, μ, Σ);
    alg=alg, inplace=inplace, apply_timechange=apply_timechange, abstol=abstol,reltol=reltol)
end

# covariance filter ODE Eqs.
compute_dP(B,P,σtil) = B*P + P*B' - outer_(σtil)
compute_dP(B,P::SArray,σtil::Number) = B*P + P*B' - σtil*σtil'*similar_type(P, Size(size(P,1),size(P,1)))(I)
compute_dν(B,ν,β::Number) = B*ν .+ β
compute_dν(B,ν,β) = B*ν + β

function compute_dP!(dP,B,P,σtil)
  dP .= B*P + P*B' - outer_(σtil)
  return nothing
end

function compute_dν!(dν,B,ν,β::Number)
  dν .= B*ν .+ β
  return nothing
end

function compute_dν!(dν,B,ν,β)
  dν .= B*ν + β
  return nothing
end

function CovarianceFilterODE(u, p, t)
  B, β, σtil = p

  # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
  ν, P, c = myunpack(u)

  # H = inv(P)
  # F = H*ν

  dν = compute_dν(B,ν,β)
  dP = compute_dP(B,P,σtil)
  dc = tr(B)

  return mypack(dν, dP, dc)
end

function CovarianceFilterODE(du, u, p, t)
  B, β, σtil = p

  # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
  ν, P, c = myunpack(u)

  #  H = inv(P)
  #  F = H*ν

  compute_dν!(du.x[1],B,ν,β)
  compute_dP!(du.x[2],B,P,σtil)
  du.x[3] .= tr(B)

  return nothing
end

function _backwardfilter(filter::CovarianceFilter,k::SDEKernel, (c, ν, P);
    alg=Euler(), inplace=false, apply_timechange=false, abstol=1e-6,reltol=1e-3)
  @unpack trange, p = k

  # Initialize OD
  u0 = mypack(ν, P, c)
  prob = ODEProblem{inplace}(CovarianceFilterODE, u0, reverse(get_tspan(trange)), p)


  if !apply_timechange
    _ts = trange # use collect() here?
  else
    _ts = timechange(trange)
  end

  if !OrdinaryDiffEq.isadaptive(alg)
    sol = solve(prob, alg, tstops=_ts, abstol=abstol, reltol=reltol)
  else
    sol = solve(prob, alg, dt=get_dt(k.trange), tstops=_ts, abstol=abstol, reltol=reltol)
  end
  message = Message(sol, k, filter, apply_timechange)

  return message, sol[end]
end

# information filter ODE Eqs.
compute_dH(H,B,σtil) = -B'*H - H*B + H*outer_(σtil)*H
compute_dF(F,H::AbstractArray,B,σtil,β::Number) = -B'*F + H*outer_(σtil)*F + H*fill(β,size(H,1))
compute_dF(F,H,B,σtil,β) = -B'*F + H*outer_(σtil)*F + H*β

function compute_dH!(dH,H,B,σtil)
  dH .= -B'*H - H*B + H*outer_(σtil)*H
  return nothing
end

function compute_dF!(dF,F,H,B,σtil,β)
  dF .= -B'*F + H*outer_(σtil)*F + H*β
  return nothing
end

function compute_dF!(dF,F,H,B,σtil,β::Number)
  dF .= -B'*F + H*outer_(σtil)*F + H*fill(β,size(H,1))
  return nothing
end

function InformationFilterODE(u, p, t)
  B, β, σtil = p

  # take care for multivariate case here if H isa Matrix, F  isa Vector, c isa Scalar
  F, H, c = myunpack(u)

  dF = compute_dF(F,H,B,σtil,β)
  dH = compute_dH(H,B,σtil)
  dc = tr(B)

  return mypack(dF, dH, dc)
end

function InformationFilterODE(du, u, p, t)
  B, β, σtil = p

  # take care for multivariate case here if H isa Matrix, F  isa Vector, c isa Scalar
  F, H, c = myunpack(u)

  #  H = inv(P)
  #  F = H*ν

  compute_dF!(du.x[1],F,H,B,σtil,β)
  compute_dH!(du.x[2],H,B,σtil)
  du.x[3] .= tr(B)

  return nothing
end

function _backwardfilter(filter::InformationFilter,k::SDEKernel, (c, F, H);
    alg=Euler(), inplace=false, apply_timechange=false, abstol=1e-6,reltol=1e-3)
  @unpack trange, p = k

  # Initialize ODE
  u0 = mypack(F, H, c)
  prob = ODEProblem{inplace}(InformationFilterODE, u0, reverse(get_tspan(trange)), p)

  if !apply_timechange
    _ts = trange # use collect() here?
  else
    _ts = timechange(trange)
  end

  if !OrdinaryDiffEq.isadaptive(alg)
    sol = solve(prob, alg, tstops=_ts, abstol=abstol, reltol=reltol)
  else
    sol = solve(prob, alg, dt=get_dt(k.trange), tstops=_ts, abstol=abstol, reltol=reltol)
  end
  message = Message(sol, k, filter, apply_timechange)

  return message, sol[end]
end


Φfunc(t,T,B) = exp(-(T-t)*B)
function solP(t,T,B,PT,Σ)
  Φ = Φfunc(t,T,B)
  Φ*(Σ+PT)*Φ'-Σ
end

function solν(t,T,B,β,νT)
  #TODO
  # β time-dependent
  νT*Φfunc(t,T,-B) + Φfunc(t,T,-B)*β*inv(B)*(Φfunc(t,T,B)-1)
end

function solc(t,T,B,cT)
  cT-tr(B)*(T-t)
end

function (G::solLyapunov)(t)
  @unpack ktilde, Σ, νT, PT, cT = G
  @unpack trange, p = ktilde
  B, β, σtil = p
  T = last(trange)

  ν = solν(t,T,B,β,νT)
  P = solP(t,T,B,PT,Σ)
  c = solc(t,T,B,cT)
  return mypack(ν, P, c)
end

function _backwardfilter(filter::LyapunovFilter,k::SDEKernel, (c, ν, P); apply_timechange=false, kwargs...)
  @unpack trange, p = k

  # solve Lyapunov equation
  B, β, σtil = p
  atil = construct_a(σtil,P)
  Σ = lyap(B,atil)

  if !apply_timechange
    _ts = trange
  else
    _ts = timechange(trange)
  end

  sol = solLyapunov(k, Σ, ν, P, c)
  soldis = hcat(sol.(_ts)...)

  message = Message(k, sol, soldis, _ts, filter)

  return message, soldis[:,1]
end
