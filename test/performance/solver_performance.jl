using Revise
import MitosisStochasticDiffEq as MSDE
using Test, Random
using LinearAlgebra
using BenchmarkTools
using MitosisStochasticDiffEq: NoiseGrid, EM

# define SDE function
f(u,p,t) = p[1]*u + p[2]
g(u,p,t) = p[3]*u

f!(du,u,p,t) = (du .= p[1]*u + p[2])
g!(du,u,p,t) = (@. du = p[3]*u)
function gstep!(dx, _, u, p, t, dw, _)
  @. dx += (p[3]*u)*dw 
end
# time span
tstart = 0.0
tend = 1.0
dt = 0.005
trange = tstart:dt:tend

B, β, σ̃ = -fill(0.1,1,1), [0.2], 1.3
p = (B, β, σ̃)

u0 = rand(1)

k_oop = MSDE.SDEKernel(f, g, trange, p)
k_iip = MSDE.SDEKernel(f!, g!, trange, p)
k_iip2 = MSDE.SDEKernel!(f!, g!, gstep!, trange, p; ws = copy(u0))


# P manually
struct customP{θType}
  θ::θType
end

function MSDE.dZ!(u, dz, Z, P::customP)
  i = u[1]
  dw = dz[3]
  @. dw = Z[i+1][3] -  Z[i][3]
  (Z[i+1][1] - Z[i][1], Z[i+1][2] - Z[i][2], dw)
end

# mul!(C, A, B, α, β) -> C = A B α + C β.

@inline function MSDE.tangent!(du, u, dz, P::customP)
  x = du[3] 
  mul!(x, P.θ[1], u[3], dz[2], false)
  @. x += P.θ[2]*dz[2] + P.θ[3]*u[3]*dz[3] 
  (dz[1], dz[2], x)
end

@inline function MSDE.exponential_map!(u, du, P::customP)
  x = u[3]
  @. x += du[3]
  (u[1] + du[1], u[2] + du[2], x)
end

# pass noise process and compare with EM()
Ws = cumsum([[zero(u0)];[sqrt(trange[i+1]-ti)*randn(size(u0))
        for (i,ti) in enumerate(trange[1:end-1])]])
NG = NoiseGrid(trange,Ws)

sol1, solend1 = @btime MSDE.sample(k_oop, u0, EM(false), NG)
sol2, solend2 = @btime MSDE.sample(k_oop, u0, MSDE.EulerMaruyama!(), Ws)
sol3, solend3 = @btime MSDE.sample(k_iip, u0, EM(false), NG)
sol4, solend4 = @btime MSDE.sample(k_iip2, u0, EM(false), NG)
sol5, solend5 = @btime MSDE.sample(k_iip2, u0, MSDE.EulerMaruyama!(), Ws)
sol6, solend6 = @btime MSDE.sample(k_iip2, u0, MSDE.EulerMaruyama!(), Ws, P=customP(p))

Z = collect(zip(1:length(trange), trange, Ws))
u = (1, 0.0, copy(u0))
dz = (0, 0.0, [0.0])
du = (0, 0.0, [0.0])
_, solend7 = MSDE.solve!(MSDE.EulerMaruyama!(), nothing, u, Z, customP(p))

@test solend1 ≈ solend2[3] rtol=1e-8
@test solend1 ≈ solend3 rtol=1e-8
@test solend1 ≈ solend4 rtol=1e-8
@test solend1 ≈ solend5[3] rtol=1e-8
@test solend1 ≈ solend6[3] rtol=1e-8
@test solend1 ≈ solend7[3] rtol=1e-8



u = (1, 0.0, copy(u0))
dz = (0, 0.0, [0.0])
@btime MSDE.solve!(MSDE.EulerMaruyama!(), nothing, deepcopy(u), Z, customP(p))
;