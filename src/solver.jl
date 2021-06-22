function zero_tangent(x, P) # euclidian fallback
  zero(x) # check false*similar
end
function zero_tangent(u::Tuple, P) # euclidian fallback
  ntuple(length(u)) do i
    false*u[i]
  end
end

function zero_integrator(Z, P)
  tozero!(deepcopy(Z[1]), P)
end

tozero!(x::Union{Number,SArray}, _) = false*x
tozero!(x::Array, _) = Base.fill!(x, false)
function tozero!(u::Tuple, P)
  ntuple(length(u)) do i
    tozero!(u[i], P)
  end
end

function dZ!(u, dz, Z, P)
  i = u[1]
  dw = dz[3]
  @. dw = Z[i+1][3] -  Z[i][3]
  (Z[i+1][1] - Z[i][1], Z[i+1][2] - Z[i][2], dw)
end


function exponential_map!(x::Array, dx, P)
  x .+= dx
end
function exponential_map!(x::Union{SArray,Number}, dx, P)
  x + dx
end

function saveit!(uu::AbstractVector, u::Tuple{Int, Float64, T}, P) where {T}
  push!(uu, (u[1], u[2], copy(u[3])))
end
saveit!(uu, u, P) = push!(uu, deepcopy(u))
saveit!(::Nothing, u, P) = nothing

endcondition(uu, u, Z, P) = u[1] >= Z[end][1]

endpoint!(u, _) = u


function solve!(solver::EulerMaruyama!, uu, u, Z, P)
  du = zero_tangent(u, P)
  dz = zero_integrator(Z, P)
  uu, u = solve_inner!(solver, uu, u, du, Z, dz, P)
  u = endpoint!(u, P)
  saveit!(uu, u, P)
  uu, u
end
function solve_inner!(solver::EulerMaruyama!, uu, u, du, Z, dz, P)
  while true
    saveit!(uu, u, P)
    dz = dZ!(u, dz, Z, P) # fetch integrator increments dz (e.g. a pair of time increment and noise increments)
    du = tangent!(du, u, dz, P) # compute time scaled stochastic tangent at current 
    u = exponential_map!(u, du, P) # move along a geodesic in tangent direction on the manifold
    endcondition(uu, u, Z, P) && break
  end
  uu, u
end
