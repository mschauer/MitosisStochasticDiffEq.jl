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
  innov(t)

t:  either an array of Numbers or StepRangeLen{Number}
returns NoiseGrid, with a Wiener process on the grid t
"""
function innov(t)
    dt = diff(t)
    w = [sqrt(dt[i])*randn(𝕏_) for i in 1:length(t)-1]
    brownian_values = cumsum(pushfirst!(w, zero(𝕏_)))
    myNoiseGrid(t,brownian_values)
end

"""
  pcn_innov(Z, ρ)

update NoiseGrid Z by pCN-step with parameter ρ. Taking ρ=1 just returns a
NoiseGrid with the same values as in Z
"""
function pcn_innov(Z, ρ)
    Znew = innov(Z.t)
    a = cumsum(pushfirst!(ρ * diff(Z.W) + sqrt(1. - ρ^2) * diff(Znew.W), zero(𝕏_)))
    myNoiseGrid(Z.t, a)
end

"""
    forwardsample(tree::Tree, rootval::𝕏, θ, dt0, f, g)

Forward sample a diffusiion process with (drift, diffusivity) = (f,g) and parameter θ on a grid with target mesh width dt0.
Starting value is rootval.

Returns Xd::Vector{𝕏} and segs, which is a vector of ODE-sols. Note that segs[1] should be skipped (not defined)
"""
function forwardsample(tree::Tree, rootval::𝕏, θ, dt0, f, g)
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
        κ = MSDE.SDEKernel(f, g, t:dt:tree.T[i], θ)
        x, xT = MSDE.sample(κ, u; save_noise=true)
        push!(Xd, xT)
        segs[i] = x
    end
    Xd, segs
end

"""
    bwfiltertree!(Q, tree::Tree, θlin, dt0; apply_time_change=false)

Backward filter on the tree, using the lienar process specified by θlin, targetting mesh-width dt0
Q must have been initialised on all leaves using a WGaussian

Returns updated Q and messages
"""
function bwfiltertree!(Q, tree::Tree, θlin, dt0; apply_time_change=false)
    T = tree.T
    messages = Vector{Any}(undef, tree.n)
    for i in reverse(eachindex(T))
        i == 1 && continue  # skip root node  (has no parent)
        ipar = tree.Par[i]

        δ = T[i]-T[ipar]
        dt = δ/round(Int, δ/dt0)

        κ̃ = MSDE.SDEKernel(MS.AffineMap(θlin[1], θlin[2]), MS.ConstantMap(θlin[3]), T[ipar]:dt:T[i], θlin)
        #        message, u = MSDE.backwardfilter(κ̃, Q[i], alg=OrdinaryDiffEq.Tsit5(), abstol=1e-12, reltol=1e-12, apply_timechange=apply_time_change)
        message, u = MSDE.backwardfilter(κ̃, Q[i], apply_timechange=apply_time_change)
        messages[i] = message
        if tree.lastone[i] #    last child is encountered first backwards
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

function fwguidtree!(X, guidedsegs, messages, tree::Tree, f, g, θ, Z; apply_time_change=false)
    ll = zeros(tree.n)
    for i in eachindex(tree.T)
        i == 1 && continue  # skip root-node (has no parent)
        κ = MSDE.SDEKernel(f, g, messages[i].ts, θ)
        ipar = tree.Par[i]
        solfw, llnew = MSDE.forwardguiding(κ, messages[i], (X[ipar], 0.0), Z[i-1]; inplace=false, save_noise=true, apply_timechange=apply_time_change)
        ll[i] = llnew + ll[ipar] * tree.lastone[i]
        X[i] = 𝕏(solfw[end][1:end-1])
        guidedsegs[i] = solfw
    end
    𝐋 = sum(ll[tree.lids])
    X, guidedsegs, ll, 𝐋
end
