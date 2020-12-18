myunpack(a) = a
myunpack(a::ArrayPartition) = (a.x[1], a.x[2], a.x[3][])
mypack(a,b,c) = ArrayPartition(a,b,[c])
mypack(a::Number...) = [a...]
mypack(a::SArray,b::SArray,c::Number) = ArrayPartition(a,b,@SVector[c])
mypack(a::SArray,b::SArray,c::SArray) = ArrayPartition(a,b,c)

function filterODE(u, p, t)
  B, β, σtil = p

  # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
  ν, P, c = myunpack(u)

#  H = inv(P)
#  F = H*ν

  dP = B*P + P*B' - σtil*σtil'
  dν = B*ν + β
  dc = tr(B)

  return mypack(dν, dP, dc)
end

function filterODE(du, u, p, t)
  B, β, σtil = p

  # take care for multivariate case here if P isa Matrix, ν  isa Vector, c isa Scalar
  ν, P, c = myunpack(u)

#  H = inv(P)
#  F = H*ν

  du.x[1] .= B*ν + β
  du.x[2] .= B*P + P*B' - σtil*σtil'
  du.x[3] .= tr(B)

  return nothing
end

function backwardfilter(k::SDEKernel, p::WGaussian{(:μ, :Σ, :c)}; alg=Euler(), inplace=false)
    message, solend = backwardfilter(k::SDEKernel, NamedTuple{(:logscale, :μ, :Σ)}((p.c, p.μ, p.Σ)); alg=alg, inplace=inplace)
    return message, WGaussian{(:μ, :Σ, :c)}(myunpack(solend)...)
end

function backwardfilter(k::SDEKernel, (c, ν, P)::NamedTuple{(:logscale, :μ, :Σ)}; alg=Euler(), inplace=false)
    @unpack tstart, tend, plin, dt = k

    trange = (tend, tstart)

    # Initialize OD
    u0 = mypack(ν, P, c)

    prob = ODEProblem{inplace}(filterODE, u0, trange, plin)
    sol = solve(prob, alg, dt=dt)
    message = sol
    return message, sol[end]
end
