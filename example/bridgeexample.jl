using LinearAlgebra, StaticArrays
I4 = SMatrix{4,4}(I)

# Define your noisy linear system
# dF(t)/dt = B*F(T) + β + σ*Noise
B = SMatrix{4,4,Float64}([-0.1 0 1.0 0 ; 0 -0.1 0 1.0; 0 0 -1 8.0; 0 0 -8.0 -1])
σ = SMatrix{4,4}(Diagonal([0.001, 0.001, 1.0, 1.0]))
β = SVector(0.0, 0.0, 0.0, 0.0)
plin = (B, β, σ)


using MitosisStochasticDiffEq, Mitosis

# define SDE function
f(u,p,t) = p[1]*u + p[2]
g(u,p,t) = diag(p[3])


# time span
tstart = 0.0
tend = 5.0
dt = 0.001

# Start point
u0 = SVector(0.0, 1.0, 1.0, 0.0)

# End point
uT = u0 # loop

sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,plin,plin,dt=dt)


WG = WGaussian{(:μ,:Σ,:c)}(uT, SMatrix{4,4}(Diagonal([0.0001, 0.0001, 0.0001, 0.0001].^(-2.0))), 0.0) # move back from endpoint plus a bit of uncertainty
WG = (;logscale = 0.0, μ=uT, Σ=SMatrix{4,4}(Diagonal([0.0000001, 0.0000001, 0.0000001, 0.0000001].^(2.0))))

message, solend = MitosisStochasticDiffEq.backwardfilter(sdekernel, WG)

solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (u0, 0.0), Z=nothing; inplace=false, save_noise=true)

u = solfw.u     
using Makie
pl = lines(getindex.(u, 1), getindex.(u, 2))
#lines!(pl, getindex.(u, 3), getindex.(u, 4), color=:blue)
#scatter!(pl, [u[1][1], u[1][3], u[end][1], u[end][3]], [u[1][2], u[1][4], u[end][2], u[end][4]]  )
#save("curve.png", pl)
pl


