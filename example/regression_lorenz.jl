import MitosisStochasticDiffEq as MSDE
using Mitosis
using Statistics
using LinearAlgebra
using StochasticDiffEq

# Estimate the parameters of a Lorenz SDE

function f(du,u,p,t)
  du[1] = p[1]*(u[2]-u[1])
  du[2] = u[1]*(p[2]-u[3]) - u[2]
  du[3] = u[1]*u[2] - p[3]*u[3]
  du
end
f(u,p,t) = f(zeros(3),u,p,t)

function g(du,u,p,t)
  du[1] = 3.0
  du[2] = 3.0
  du[3] = 3.0
  du
end
g(u,p,t) = g(zeros(3),u,p,t)


# create solution with true parameters
p = [10.0,28.0, 8/3]

u0 = [1.0,0.0,0.0]
tspan = (0.0,10.0)
dt = 0.01
trange = tspan[1]:dt:tspan[2]

prob = SDEProblem(f, g, u0, tspan, p)
sol = solve(prob, EM(false), dt = dt)


# Define Jacobians

# f is linear in the parameters with Jacobian
function f_jac(J,u,p,t)
    J .= false
    J[1,1] = (u[2]-u[1])
    J[2,2] = u[1]
    J[3,3] = -u[3]
    nothing
end
# and intercept
function ϕ0(du,u,p,t)
    du[1] = false
    du[2] = u[1]*(-u[3]) - u[2]
    du[3] = u[1]*u[2]
end


# Solve the regression problem
sdekernel = MSDE.SDEKernel(f,g,trange,p)

ϕprototype = zeros((3,3)) # prototypes for vectors
yprototype = zeros((3,))
R = MSDE.Regression!(sdekernel,
  yprototype,paramjac_prototype=ϕprototype,paramjac=f_jac,intercept=ϕ0)
G = MSDE.conjugate(R, sol, 0.1*I(3))

# Estimate
p̂ = mean(G)
se = sqrt.(diag(cov(G)))
display(map((p̂, se, p) -> "$(round(p̂, digits=2)) ± $(round(se, digits=2)) (true: $p)", p̂, se, p))


# Plots
using Plots, LaTeXStrings
pl1 = plot(sol,vars=(1,2,3), legend=true,
  #background_color = :Transparent,
  label = "",
  lw = 2,
  xlabel = L"x", ylabel = L"y", zlabel = L"z",
  size=(350,300),
  labelfontsize=20
 )
# savefig(pl1, "Lorentz.png")
