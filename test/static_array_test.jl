using MitosisStochasticDiffEq
using Test, Random
using StaticArrays
using Statistics


seed = 1234
Random.seed!(seed)

# define SDE function
f(u,p,t) = p[1]*u .+ p[2]
g(u,p,t) = p[3]

function fstat(u,p,t)
  dx = p[1]*u[1] + p[2]
  @SVector [dx]
end

function gstat(u,p,t)
  @SVector [p[3]]
end

# time span
tstart = 0.0
tend = 1.0
dt = 0.01

# initial condition
u0 = [1.1]
u0stat = @SVector [1.1]

# set true model parameters
par = [-0.2, 0.1, 0.9]

# set of linear parameters Eq.~(2.2)
plin = copy(par)
pest = copy(par)

sdekernel1 = MitosisStochasticDiffEq.SDEKernel(f,g,tstart,tend,pest,plin,dt=dt)
sdekernel2 = MitosisStochasticDiffEq.SDEKernel(fstat,gstat,tstart,tend,pest,plin,dt=dt)

Random.seed!(seed)
samples1 = MitosisStochasticDiffEq.sample(sdekernel1, u0, save_noise=false)
Random.seed!(seed)
samples2 = MitosisStochasticDiffEq.sample(sdekernel2, u0stat, save_noise=false)

@test isapprox(samples1[2][1], samples2[2][1], rtol=1e-10)
@test typeof(samples2[2]) <: SArray


# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues1 =  [0.0, 1.5, 0.1];
NT1 = NamedTuple{mynames}(myvalues1)

#myvalues2 = @SVector [0.0, 1.5, 0.1] # will be just converted to a standard array
myvalues2 = [(@SVector [0.0]), (@SVector [1.5]), (@SVector[0.1]) ]
NT2 = NamedTuple{mynames}(myvalues2)

message1, backward1 = MitosisStochasticDiffEq.backwardfilter(sdekernel1, NT1)
message2, backward2 = MitosisStochasticDiffEq.backwardfilter(sdekernel2, NT2)

@test isapprox(backward1, backward2, rtol=1e-10)
@test typeof(backward2.x[1]) <: SArray



x0 = [1.34]
ll0 = randn()

Random.seed!(seed)
samples1 = MitosisStochasticDiffEq.forwardguiding(sdekernel1, message1, (x0, ll0), inplace=true)
Random.seed!(seed)
samples2 = MitosisStochasticDiffEq.forwardguiding(sdekernel2, message2, (x0, ll0), inplace=false)

@test isapprox(samples1[2][1], samples2[2][1], rtol=1e-10)
@test_broken typeof(samples2[1][end]) <: SArray
