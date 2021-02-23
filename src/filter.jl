myunpack(a) = a
myunpack(a::ArrayPartition) = (a.x[1], a.x[2], a.x[3][])
mypack(a,b,c) = ArrayPartition(a,b,[c])
mypack(a::Number...) = [a...]
mypack(a::SArray,b::SArray,c::Number) = ArrayPartition(a,b,@SVector[c])
mypack(a::SArray,b::SArray,c::SArray) = ArrayPartition(a,b,c)


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

function filterODE(u, p, t)
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

function filterODE(du, u, p, t)
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

function backwardfilter(k::SDEKernel, p::WGaussian{(:μ, :Σ, :c)}; alg=Euler(), inplace=false)
  message, solend = backwardfilter(k::SDEKernel, NamedTuple{(:logscale, :μ, :Σ)}((p.c, p.μ, p.Σ)); alg=alg, inplace=inplace)
  return message, WGaussian{(:μ, :Σ, :c)}(myunpack(solend)...)
end

function backwardfilter(k::SDEKernel, (c, ν, P)::NamedTuple{(:logscale, :μ, :Σ)}; alg=Euler(), inplace=false)
  @unpack trange, p = k

  # Initialize OD
  u0 = mypack(ν, P, c)

  prob = ODEProblem{inplace}(filterODE, u0, reverse(get_tspan(trange)), p)
  sol = solve(prob, alg, dt = get_dt(trange))
  message = Message(sol, k)
  return message, sol[end]
end
