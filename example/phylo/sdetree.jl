
function MSDE.dZ!(u, dz::Tuple{Int64, Float64, <:SVector}, Z, P::MSDE.SDEKernel)
    i = u[1]
    (Z[i+1][1] - Z[i][1], Z[i+1][2] - Z[i][2],  Z[i+1][3] -  Z[i][3])
end

wrapdiagonal(x) = x
wrapdiagonal(x::Vector) = Diagonal(x)

function MSDE.tangent!(du::Tuple{Int64, Float64, <:SVector}, u, dz, P::MSDE.SDEKernel)
    MSDE.@unpack f, g, p, noise_rate_prototype = P
    k1 = f(u[3],p,u[2])
    g1 = wrapdiagonal(g(u[3],p,u[2]))
    du3 = k1*dz[2] + g1*dz[3]
    (dz[1], dz[2], du3)
end

function MSDE.exponential_map!(u::Tuple{Int64, Float64, <:SVector}, du, P::MSDE.SDEKernel)
    (u[1] + du[1], u[2] + du[2], u[3] + du[3])
end



function myNoiseGrid(t,W,Z=nothing;reset=true)
  val = W[1]
  curt = t[1]
  dt = t[1]
  curW = copy(val)
  dW = copy(val)
  if Z==nothing
    curZ = nothing
    dZ = nothing
  else
    curZ = copy(Z[1])
    dZ = copy(Z[1])
  end
  DiffEqNoiseProcess.NoiseGrid{typeof(val),ndims(val),typeof(dt),typeof(dW),typeof(dZ),typeof(Z),false}(
            t,W,W,Z,curt,curW,curZ,dt,dW,dZ,true,reset)
end

"""
  myinnov(t)

t:  either an array of Numbers or StepRangeLen{Number}
returns NoiseGrid, with a Wiener process on the grid t.
fac: factor that allows to scale the noise values (e.g., set them to zero)
"""
function myinnov(t, 𝕏_ ,fac=1)
    dt = diff(t)
    w = [fac*sqrt(dt[i])*randn(𝕏_) for i in 1:length(t)-1]
    brownian_values = cumsum(pushfirst!(w, zero(𝕏_)))
    myNoiseGrid(t,brownian_values)
end


"""
  pcn_innov(Z, ρ)

update NoiseGrid Z by pCN-step with parameter ρ. Taking ρ=1 just returns a
NoiseGrid with the same values as in Z
"""
function pcn_innov(Z, ρ, 𝕏_, fac=1)
    Znew = myinnov(Z.t, 𝕏_, fac)
    a = cumsum(pushfirst!(ρ * diff(Z.W) + sqrt(1. - ρ^2) * diff(Znew.W), zero(𝕏_)))
    myNoiseGrid(Z.t, a)
end

"""
    forwardsample(tree::Tree, rootval::𝕏, θ, dt0, f, g)

Forward sample a diffusiion process with (drift, diffusivity) = (f,g) and parameter θ on a grid with target mesh width dt0.
Starting value is rootval.

Returns Xd::Vector{𝕏} and segs, which is a vector of ODE-sols. Note that segs[1] should be skipped (not defined)
"""
#function forwardsample(tree::Tree, rootval::𝕏, θ, dt0, f, g)
function forwardsample(tree::Tree, rootval, θ, dt0, f, g, SDEalg)
    Xd = [rootval]    # save endpoints
    segs = Vector{Any}(undef, tree.n) # save all guided segments
    for i in eachindex(tree.T)
        #local κ, x, xT, dt, u
        i == 1 && continue  # skip root-node  (has no parent)

        # define timegrid on segment
        i′ = tree.Par[i]
        t = tree.T[i′]
        m = round(Int, (tree.T[i]-t)/dt0)
        dt = (tree.T[i] - t)/m

        u = Xd[i′]   # starting value
        # set noise_rate_prototype for comparison with StochasticDiffEq package
        κ = MSDE.SDEKernel(f, g, t:dt:tree.T[i], θ, Diagonal(θ[2]))
        # choose SDEalg =  MSDE.EulerMaruyama!() for new fast solver...
        NG = myinnov(t:dt:tree.T[i], 𝕏)
        xT, seg  = MSDE.sample(κ, u, SDEalg, NG)
        push!(Xd, xT)
        segs[i] = seg
    end
    Xd, segs
end

"""
    bwfiltertree!(Q, tree::Tree, θlin, dt0; apply_time_change=false)

Backward filter on the tree, using the lienar process specified by θlin, targetting mesh-width dt0
Q must have been initialised on all leaves using a WGaussian

Returns updated Q and messages
"""
function bwfiltertree!(Q, tree::Tree, θlin, dt0; apply_time_change=false, alg=Tsit5())
    T = tree.T
    messages = Vector{Any}(undef, tree.n)
    for i in reverse(eachindex(T))
        i == 1 && continue  # skip root node  (has no parent)
        ipar = tree.Par[i]

        δ = T[i]-T[ipar]
        dt = δ/round(Int, δ/dt0)
        if apply_time_change
            tvals = MSDE.timechange(T[ipar]:dt:T[i])
        else
            tvals = T[ipar]:dt:T[i]
        end

        κ̃ = MSDE.SDEKernel(MS.AffineMap(θlin[1], θlin[2]), MS.ConstantMap(θlin[3]), tvals, θlin)

        # message, u = MSDE.backwardfilter(κ̃, Q[i], alg=OrdinaryDiffEq.Tsit5(), abstol=1e-12, reltol=1e-12, apply_timechange=apply_time_change)
        message, u = MSDE.backwardfilter(κ̃, Q[i], apply_timechange=apply_time_change, alg=alg)
        messages[i] = message
        if tree.lastone[i] # last child is encountered first backwards
            Q[ipar] = u
        else
            Q[ipar] = MS.fuse(u, Q[ipar]; unfused=false)[2]
        end
    end
    Q, messages
end

"""
    fwguidtree!(X, guidedsegs, messages, tree::Tree, f, g, θ, Z; apply_time_change=false)

Returns updated `X` and `guidedsegs`, as also the loglikelihood vector `ll` (at all vertices of the tree) and
`𝐋` the loglikelihood summed over all leaf-indices.
"""

function fwguidtree!(X, guidedsegs, Q, messages, tree::Tree, f, g, θ, Z, SDEalg; apply_time_change=false)
    ll = zeros(tree.n)
    for i in eachindex(tree.T)
        i == 1 && continue  # skip root-node (has no parent)
        κ = MSDE.SDEKernel(f, g, messages[i].ts, θ, zeros(d,d))
        ipar = tree.Par[i]
        (solend, llnew), res = MSDE.forwardguiding(κ, messages[i], (X[ipar], 0.0), SDEalg, Z[i-1],
                                                            inplace=false, apply_timechange=apply_time_change)
        ll[i] = llnew + ll[ipar] * tree.lastone[i]
        X[i] = solend
        guidedsegs[i] = res
    end
    𝐋 = sum(ll[tree.lids]) + logdensity(Q[1], X[1]) #logdensity(convert(WGaussian{(:F,:Γ,:c)},Q[1]), X[1])
    X, guidedsegs, ll, 𝐋
end
