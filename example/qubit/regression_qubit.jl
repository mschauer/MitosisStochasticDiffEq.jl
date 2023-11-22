using Pkg
path = @__DIR__
cd(path)
Pkg.activate(".")
Pkg.instantiate()

using Test, Random
include("qubit.jl")

Δ = 3.0
Ω = 1.0
κ = 1.0

p = [Δ, Ω, κ]

dt = 0.001
tspan = (0.0,10.0)
trange = tspan[1]:dt:tspan[2]

##
# single trajectory approach

# set scalar random process based on noise grid
noise = scalar_noise(dt, trange)
u0 = prepare_initial()


proboop = SDEProblem{false}(qubit_drift, qubit_diffusion, u0, tspan, p,
         noise=noise
      )

sol = solve(proboop, EM(), dt=dt)

yprototype = zeros(4)
ϕprototype = zeros((4,2)) # Δ and Ω can be inferred in this example "without" controller

R = MSDE.Regression{false}(proboop,paramjac=f_jac,intercept=ϕ0, m=1)
G = MSDE.conjugate(R, sol, 0.1*I(2))

# Estimate
p̂ = mean(G)
se = sqrt.(diag(cov(G)))
display(map((p̂, se, p) -> "$(round(p̂, digits=3)) ± $(round(se, digits=3)) (true: $p)", p̂, se, [Δ,Ω]))
@test p̂[1] ≈ Δ rtol=se[1]
@test p̂[2] ≈ Ω rtol=se[2]

##

##
# Simulate path ensemble 
u0_x = [1/sqrt(2),1/sqrt(2),0.0,0.0]

# set scalar random process
seed = 123
Random.seed!(seed)
W = WienerProcess(0.0,0.0,0.0)

prob = SDEProblem{false}(qubit_drift, qubit_diffusion, u0_x, tspan, p,
         noise=W
      )
function prob_func(prob, i, repeat)
  W = WienerProcess(0.0,0.0,0.0)
  remake(prob,noise=W)
end      
ensembleprob = EnsembleProblem(prob, prob_func=prob_func)
ensemblesol = Array(solve(
    ensembleprob, EM(), EnsembleThreads(); dt=dt, trajectories=1000
))

ceRs =  @view ensemblesol[1,:,:]
cdRs =  @view ensemblesol[2,:,:]
ceIs =  @view ensemblesol[3,:,:]
cdIs =  @view ensemblesol[4,:,:]

using Plots
xmean = mean(2*(ceRs.*cdRs+ceIs.*cdIs)./(ceRs.^2 + cdRs.^2 + ceIs.^2 + cdIs.^2), dims=2)[:]
xstd = std(2*(ceRs.*cdRs+ceIs.*cdIs)./(ceRs.^2 + cdRs.^2 + ceIs.^2 + cdIs.^2), dims=2)[:]
ymean = mean(2*(ceRs.*cdIs-ceIs.*cdRs)./(ceRs.^2 + cdRs.^2 + ceIs.^2 + cdIs.^2), dims=2)[:]
ystd = std(2*(ceRs.*cdIs-ceIs.*cdRs)./(ceRs.^2 + cdRs.^2 + ceIs.^2 + cdIs.^2), dims=2)[:]
zmean = mean((ceRs.^2+ceIs.^2-cdRs.^2-cdIs.^2)./(ceRs.^2 + cdRs.^2 + ceIs.^2 + cdIs.^2), dims=2)[:]
zstd = std((ceRs.^2+ceIs.^2-cdRs.^2-cdIs.^2)./(ceRs.^2 + cdRs.^2 + ceIs.^2 + cdIs.^2), dims=2)[:]
plx = plot(trange, xmean, ribbon=xstd)
ply = plot(trange, ymean, ribbon=ystd)
plz = plot(trange, zmean, ribbon=zstd)

pl = plot(plx,ply,plz)

# Inference on drift parameters
sdekernel = MSDE.SDEKernel(qubit_drift,qubit_diffusion,trange,0*p)
ϕprototype = zeros((length(u0_x),length(p))) # prototypes for vectors
yprototype = zeros((length(u0_x),))
R = MSDE.Regression!(sdekernel,yprototype,paramjac_prototype=ϕprototype,paramjac=f_jac!,intercept=ϕ0!)
prior_precision = 0.1I(3)
posterior = MSDE.conjugate(R, ensemblesol, prior_precision)
print(mean(posterior), " ± ", sqrt(cov(posterior)))