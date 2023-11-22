import MitosisStochasticDiffEq as MSDE
using Mitosis
using Statistics
using LinearAlgebra
using StochasticDiffEq
using DiffEqNoiseProcess


# non-mutating definitions
function qubit_drift(u,p,t)
  # expansion coefficients |Ψ> = ce |e> + cd |d>
  ceR, cdR, ceI, cdI = u # real and imaginary parts

  # norm for expectation values
  norm = ceR^2 + cdR^2 + ceI^2 + cdI^2

  # unpack controller specifications
  Δ, Ω, κ = p

  # Δ: atomic frequency
  # Ω: Rabi frequency for field in x direction
  # κ: spontaneous emission

  du1 = -(cdI*ceI + cdR*ceR)^2*κ*u/(2*norm^2)

  dx1 = 1//2*(ceI*Δ-ceR*κ+cdI*Ω)
  dx2 = -cdI*Δ/2 + 1*ceR*(cdI*ceI+cdR*ceR)*κ/norm + ceI*Ω/2
  dx3 = 1//2*(-ceR*Δ-ceI*κ-cdR*Ω)
  dx4 = cdR*Δ/2 + 1*ceI*(cdI*ceI+cdR*ceR)*κ/norm - ceR*Ω/2

  du2 = [dx1,dx2,dx3,dx4]
  return du1 + du2
end

function qubit_diffusion(u,p,t)
  ceR, cdR, ceI, cdI = u # real and imaginary parts

  κ = p[end]

  # norm for expectation values
  norm = ceR^2 + cdR^2 + ceI^2 + cdI^2

  du1 = -1*(cdI*ceI + cdR*ceR)*sqrt(κ)*u/norm
  du2 = [zero(ceR), sqrt(κ)*ceR, zero(ceR), sqrt(κ)*ceI]
  return du1+du2
end

# initial state anywhere on the Bloch sphere
function prepare_initial()
  # random position on the Bloch sphere
  theta = acos(2*rand()-1)  # uniform sampling for cos(theta) between -1 and 1
  phi = rand()*2*pi  # uniform sampling for phi between 0 and 2pi
  # real and imaginary parts ceR, cdR, ceI, cdI
  u0 = [cos(theta/2), sin(theta/2)*cos(phi), false*theta, sin(theta/2)*sin(phi)]
  return u0
end


function f_jac(u,p,t)
  ceR, cdR, ceI, cdI = u # real and imaginary parts

  [ ceI/2      cdI/2
    -cdI/2     ceI/2
    -ceR/2    -cdR/2
    cdR/2     -ceR/2 ]
end

function f_jac!(du,u,p,t)
  ceR, cdR, ceI, cdI = u # real and imaginary parts

  du[1,1] = ceI/2
  du[1,2] = cdI/2
  du[2,1] = -cdI/2
  du[2,2] = ceI/2
  du[3,1] = -ceR/2
  du[3,2] = -cdR/2
  du[4,1] = cdR/2
  du[4,2] = -ceR/2
  return nothing
end

# intercept
function ϕ0(u,p,t)
  ceR, cdR, ceI, cdI = u # real and imaginary parts
  # norm for expectation values
  norm = ceR^2 + cdR^2 + ceI^2 + cdI^2
  du1 = -(cdI*ceI + cdR*ceR)^2*κ*u/(2*norm^2)

  dx1 = -1//2*ceR*κ
  dx2 = 1*ceR*(cdI*ceI+cdR*ceR)*κ/norm
  dx3 = -1//2*ceI*κ
  dx4 = 1*ceI*(cdI*ceI+cdR*ceR)*κ/norm

  du2 = [dx1,dx2,dx3,dx4]

  return du1 + du2
end

function ϕ0!(du,u,p,t)
  ceR, cdR, ceI, cdI = u # real and imaginary parts
  # norm for expectation values
  norm = ceR^2 + cdR^2 + ceI^2 + cdI^2
  @. du = -(cdI*ceI + cdR*ceR)^2*κ*u/(2*norm^2)

  du[1] += -1//2*ceR*κ
  du[2] +=  1*ceR*(cdI*ceI+cdR*ceR)*κ/norm
  du[3] += -1//2*ceI*κ
  du[4] +=  1*ceI*(cdI*ceI+cdR*ceR)*κ/norm

  return nothing
end


function scalar_noise(dt, ts)
  W = sqrt(dt)*randn(typeof(dt),size(ts)) #for 1 trajectory
  W1 = cumsum([zero(dt); W[1:end-1]], dims=1)
  return NoiseGrid(ts,W1)
end
