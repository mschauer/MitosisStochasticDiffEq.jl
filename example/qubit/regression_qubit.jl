using Pkg
path = @__DIR__
cd(path)
Pkg.activate(".");Pkg.instantiate()

include("qubit.jl")

Δ = 20.0
Ω = 10.0
κ = 1.0

p = [Δ, Ω, κ]

dt = 0.001
tspan = (0.0,1.0)
trange = tspan[1]:dt:tspan[2]

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
