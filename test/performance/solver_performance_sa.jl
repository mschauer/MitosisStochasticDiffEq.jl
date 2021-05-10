using Revise
import MitosisStochasticDiffEq as MSDE
using Test, Random
using LinearAlgebra
using BenchmarkTools
using MitosisStochasticDiffEq: NoiseGrid, EM
using Mitosis: WGaussian
using StaticArrays

#=
function MSDE.saveit!(uu::Vector, u::Tuple{Int64, Float64, Vector{Float64}}, P)
  push!(uu, (u[1], u[2], copy(u[3])))
end
=#

# define SDE function
f(u,p,t) = p[1]*u + p[2]
g(u,p,t) = p[3]

# time span
tstart = 0.0
tend = 1.0
dt = 0.0001
trange = tstart:dt:tend

B, β, σ̃ = @SMatrix([-0.1]), @SVector([0.2]), 1.3
p = (B, β, σ̃)

u0 = @SVector [1.0]

k_oop = MSDE.SDEKernel(f, g, trange, p)




# pass noise process and compare with EM()
w = zero(u0)
Ws = [w]
for (i,ti) in enumerate(trange[1:end-1])
    push!(Ws, w + sqrt(trange[i+1]-ti)*@SVector(randn(1)))
end

function MSDE.dZ!(u, dz::Tuple{Int64, Float64, <:SVector}, Z, P::MSDE.SDEKernel)
    i = u[1]
    (Z[i+1][1] - Z[i][1], Z[i+1][2] - Z[i][2],  Z[i+1][3] -  Z[i][3])
end

function MSDE.tangent!(du::Tuple{Int64, Float64, <:SVector}, u, dz, P::MSDE.SDEKernel)
    MSDE.@unpack f, g, p, noise_rate_prototype = P
    k1 = f(u[3],p,u[2])
    g1 = g(u[3],p,u[2])
    du3 = k1*dz[2] + g1*dz[3]
    (dz[1], dz[2], du3)
end

function MSDE.exponential_map!(u::Tuple{Int64, Float64, <:SVector}, du, P::MSDE.SDEKernel)
    (u[1] + du[1], u[2] + du[2], u[3] + du[3])
end

sol2, solend2 = @btime MSDE.sample(k_oop, u0, MSDE.EulerMaruyama!(), Ws)
v = solend2[end]

Z = collect(zip(1:length(trange), trange, Ws))
u = (1, 0.0, u0)

sol7, solend7 = @time MSDE.solve!(MSDE.EulerMaruyama!(), typeof(u)[], u, Z, k_oop)
@test solend2[3] ≈ solend7[3] rtol=1e-8

@btime MSDE.solve!(MSDE.EulerMaruyama!(), nothing, u, Z, k_oop);


message1_, backward1 = MSDE.backwardfilter(k_oop, WGaussian{(:μ, :Σ, :c)}(v, @SMatrix([1.0]), 0.0))
U = reinterpret(Tuple{SVector{1, Float64}, SMatrix{1, 1, Float64, 1}, Float64}, message1_.soldis)
message1 = MSDE.Message(message1_.ktilde, message1_.sol, U, message1_.ts, message1_.filter)
gp1 = MSDE.GuidedSDE(k_oop, message1)


ξ = (1, 0.0, u0, 0.0)
sol, solend = MSDE.solve!(MSDE.EulerMaruyama!(), typeof(ξ)[], ξ, Z, gp1)

@btime MSDE.solve!(MSDE.EulerMaruyama!(), nothing, ξ, Z, gp1);
