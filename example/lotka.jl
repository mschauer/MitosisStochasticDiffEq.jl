import MitosisStochasticDiffEq as MSDE
using Mitosis
using Statistics
using LinearAlgebra
using StochasticDiffEq

# Estimate the parameters of a Lorenz SDE

u0 = [1.0,1.0]
tspan = (0.0,10.0)
function multiplicative_noise!(du,u,p,t)
  x,y = u
  du[1] = p[5]*x
  du[2] = p[6]*y
end
p = ptrue = [1.5,1.0,3.0,1.0,0.1,0.1]

function lotka_volterra!(du,u,p,t)
  x,y = u
  α,β,γ,δ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = δ*x*y - γ*y
end

# define log.(u) via Ito

ξ0 = log.(u0)
function g!(du,u,p,t)
  x,y = u
  du[1] = p[5]
  du[2] = p[6]
  nothing
end

function f!(du,u,p,t)
  x,y = u
  α,β,γ,δ = p
  du[1] = dx = α - β*exp(y)
  du[2] = dy = δ*exp(x) - γ
  nothing
end
function f(u, p, t) 
  du = zeros(2)
  f!(du, u, p, t)
  du
end


# f is linear in the parameters with Jacobian
function f_jac(J,u,p,t)
  x,y = u
  J .= false
  J[1,1] = 1.0
  J[1,2] = -exp(y)
  J[2,3] = -1.0
  J[2,4] = exp(x)
  nothing
end
# and intercept
function ϕ0(du,u,p,t)
  du .= false
  nothing
end


using Plots

prob_sde = SDEProblem(lotka_volterra!,multiplicative_noise!,u0,tspan,p)
ensembleprob = EnsembleProblem(prob_sde)
@time data = solve(ensembleprob,SOSRI(),saveat=0.1,trajectories=1000)
plot(EnsembleSummary(data))


prob_sde2 = SDEProblem(f!,g!,ξ0,tspan,p)
ensembleprob2 = EnsembleProblem(prob_sde2)
data2 = solve(ensembleprob2,SOSRI(),saveat=0.1,trajectories=1000)

@time data2exp = solve(ensembleprob2,SOSRI(),saveat=0.1,trajectories=1000)
for u in data2exp.u
  u.u .= [exp.(x) for x in u.u]
end

plot(EnsembleSummary(data2exp))

quvar(u) = sum((du - f(u, p, t)*dt).^2 for (t,u, dt, du) in zip(u.t, u.u, diff(u.t), diff(u.u)))



p = (0.5 .+ 1.5rand(length(ptrue))).*ptrue
ps = [copy(p)]
for k in 1:5
  global p
  # Solve the regression problem

  sdekernel = MSDE.SDEKernel(f!,g!,tspan,p)

  ϕprototype = zeros((2,4)) # prototypes for vectors
  yprototype = zeros((2,))
  R = MSDE.Regression!(sdekernel,yprototype,paramjac_prototype=ϕprototype,paramjac=f_jac,intercept=ϕ0)

  G = MSDE.conjugate(R, data2, 0.1I(4))

  # Estimate
  p̂ = mean(G)
  se = sqrt.(diag(cov(G)))
  display(map((p̂, se, p) -> "$(round(p̂, digits=2)) ± $(round(se, digits=2)) (true: $p)", p̂, se, ptrue))

  p[1:4] = rand(convert(Gaussian{(:μ, :Σ)}, G))

  # Estimate covariance
  T = sum(u.t[end] - u.t[1] for u in data2)

  Q = sum(quvar(u) for u in data2)
  
  p[end-1:end] = sqrt.(Q/T)
  push!(ps, copy(p))
end 
ps
its = eachindex(ps)
cols = [k for j in its, k in eachindex(ptrue)]
pl = scatter(reduce(hcat, ps)', c=cols, labels=["α" "β" "γ" "δ" "s1" "s2"])
plot!(reduce(hcat, fill(ptrue, length(ps)))', c=cols, labels=["αtrue" "βtrue" "γtrue" "δtrue" "s1true" "s2true"])