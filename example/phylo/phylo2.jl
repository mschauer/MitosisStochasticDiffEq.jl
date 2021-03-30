using Pkg
path = @__DIR__
cd(path)
Pkg.activate(path); Pkg.instantiate()
using Mitosis
using MitosisStochasticDiffEq
using LinearAlgebra, Statistics, Random, StatsBase
using DelimitedFiles
using StaticArrays
using Makie
using NewickTree
using OrdinaryDiffEq
using DiffEqNoiseProcess

const d = 2
const 𝕏 = SVector{d,Float64}
const MS = Mitosis
const MSDE = MitosisStochasticDiffEq


## define model (two traits, as in presentation)
θ = ([3.0, 1.2], [0.1, 0.1])
const M = -0.05I + [-1.0 1.0; 1.0 -1.0]

B(θ) = Diagonal(θ[1]) * M
Σ(θ) = Diagonal(θ[2])

f(u,θ,t) = tanh.(Diagonal(θ[1]) * M * u)
g(u,θ,t) = θ[2]

θlin = (B(θ), zeros(d), Σ(θ))

## tree
struct Tree{newicktype, Teltype, Pareltype, idseltype, nameseltype}
    newick::newicktype   #   (:id, :data, :parent, :children)
    T::Vector{Teltype}
    Par::Vector{Pareltype}
    ids::Vector{idseltype}
    names::Vector{nameseltype}
    lids::Vector{idseltype}
    lnames::Vector{nameseltype}
    lastone::Vector{Bool} # true if it is the last child when running forwards
    n::Int64
end

function Tree(S::String)
    nwtree = readnw(S)

    ids = map(x->x.id, prewalk(nwtree))
    names = map(x->x.data.name, prewalk(nwtree))
    n = length(ids)
    # leaves
    l = getleaves(nwtree)
    lids = map(x -> Int(x.id), l)
    lnames = map(x->x.data.name, l)
    #@assert issubset(tipslist[:,1], lnames)

    # extract parents and time spans (on segments)
    Par = zeros(Int, n)  # parent ids
    T = zeros(n)  # time spans
    for node in prewalk(nwtree)
        isroot(node) && continue
        p = node.parent.id
        Par[node.id] = p
        T[node.id] = T[p] + node.data.distance
    end
    @assert all(Par .< 1:n)

    lastone = zeros(Bool,n)
    parentdone = zeros(Bool,n)
    for i ∈ n:-1:2
        if !parentdone[Par[i]]
            lastone[i] = true
            parentdone[Par[i]] = true
        end
    end
    Tree{typeof(nwtree), eltype(T), eltype(Par), eltype(ids), eltype(names)}(nwtree,  T, Par, ids, names, lids, lnames,lastone, n)
end





## read tree (very simple example)
S = "(A:0.1,B:0.2,(C:0.3,D:0.4)E:0.5)F;"
S20 = "(((t20:0.1377639426,t8:0.3216105914):0.461882598,(((t14:0.09650261351,t12:0.5344695284):0.1058950284,(t1:0.7008816295,t13:0.8587384557):0.7271672823):0.8026451592,(t6:0.5240857233,(t16:0.451590274,t10:0.4654036982):0.8569358122):0.140687139):0.7207935331):0.06727131596,((((t17:0.03115223325,t18:0.8930883685):0.6862687438,t9:0.4896655723):0.3056753092,t3:0.2549535078):0.07862622454,(((t4:0.03973490326,t5:0.4096179632):0.3020580804,t15:0.06550096255):0.2660011558,(t7:0.3895205685,((t19:0.165174433,t2:0.6469203965):0.2730371242,t11:0.2700119715):0.8393620371):0.02117527928):0.206376995):0.6012785432);"
S50 = "((((((t23:0.5766009684,t33:0.9917489749):0.3572315329,(t40:0.4535627225,t25:0.1296473157):0.9540394924):0.4740875189,(t6:0.946340692,t47:0.386682119):0.1535291784):0.07034701738,((((t49:0.6612032412,t27:0.1606025242):0.1527153626,t15:0.1854858669):0.5782065149,t50:0.3087045723):0.0759275395,((t17:0.258283512,t35:0.7651973176):0.2003453802,t2:0.4033703087):0.2418845498):0.6834833182):0.4268306631,(((t26:0.1744227773,(t16:0.6275201498,t10:0.09013780276):0.1857208994):0.8804157388,(t28:0.626011228,t46:0.8026940082):0.7066663445):0.6334459039,(t45:0.3075582262,t42:0.2759609232):0.05279792193):0.4003514573):0.5333229457,((((t30:0.05955170072,t41:0.2500256761):0.04172409023,((t19:0.3332289669,t31:0.6650944015):0.2450174955,t9:0.3372890623):0.2041792406):0.3337947708,t1:0.9396123393):0.0896910606,(((((t12:0.1379979164,t43:0.9007850592):0.703310129,(t11:0.3347885907,(t22:0.7190679635,t14:0.8988074451):0.1574828231):0.9644375315):0.137505424,((t37:0.7556577872,t8:0.3481238757):0.7712922962,(t7:0.7946650374,t48:0.8638419164):0.338934626):0.7507952554):0.5193078597,(t4:0.2408366969,(((t29:0.7862726359,t38:0.4319011718):0.8655222375,((t20:0.09156441619,t3:0.5975058537):0.3240908289,(t34:0.7324223334,t32:0.1348467385):0.4636785307):0.7708284601):0.7991992962,t39:0.2996040138):0.4087550903):0.1688272723):0.7225795928,(((t44:0.02794037666,t5:0.09033886855):0.5778779227,(t24:0.8744501146,((t21:0.2940180032,t18:0.03399693617):0.8231791619,t13:0.5192477896):0.5669933448):0.9547001948):0.8561708115,t36:0.9225172002):0.02867545444):0.1595393196):0.1254606678);"

tree = Tree(S20)
print_tree(tree.newick)
# fieldnames(typeof(nwtree))

# # extract name, id
# ids = map(x->x.id, prewalk(nwtree))
# nams = map(x->x.data.name, prewalk(nwtree))
# n = length(ids)
# @assert all(ids .== 1:length(ids))
#

#=
ids # Vertex ids
nams # Vertex names
X # vertex values
lids # leaf ids
lnames # leaf names
Par # Parent ids, parents of first node set to 0
T # Times
@assert all(T[Par[ids][2:end]] .< T[ids[2:end]])
=#

## forward sample on the tree to simulate the observations at the tips

dt0 = 0.001
u0 = zero(𝕏)  # value at root node
Xd = [u0]    # save endpoints
segs = Vector{Any}(undef, tree.n) # save all guided segments

for i in eachindex(tree.T)
    local κ, x, xT, dt
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

## make dictionary of the tips (leaves), no values as yet
# tipsdict = Dict()
# for leaf in lnames
#     push!(tipsdict, leaf => 𝕏(zeros(d)))
# end





## backward filter, observations are in X

function bwfiltertree!(Q, tree::Tree, θlin, dt0)
    # backward filter
    T = tree.T
    messages = Vector{Any}(undef, tree.n)  # save all messages
    for i in reverse(eachindex(T))
        i == 1 && continue  # skip root node  (has no parent)
        i′ = tree.Par[i]

        m = round(Int, (T[i]-T[i′])/dt0)
        dt = (T[i] - T[i′])/m

        κ̃ = MSDE.SDEKernel(MS.AffineMap(θlin[1], θlin[2]), MS.ConstantMap(θlin[3]), T[i′]:dt:T[i], θlin)
        #        message, u = MSDE.backwardfilter(κ̃, Q[i], alg=OrdinaryDiffEq.Tsit5(), abstol=1e-12, reltol=1e-12, apply_timechange=true)
        message, u = MSDE.backwardfilter(κ̃, Q[i], apply_timechange=true)
        messages[i] = message

    #    @assert ismissing(Q[i′]) == lastone[i]
        if tree.lastone[i] #last child is encountered first backwards
            Q[i′] = u
        else
            Q[i′] = MS.fuse(u, Q[i′]; unfused=false)[2]
        end
    end
    Q, messages
end



## simulate conditioned process using messages

function fwguidtree!(messages, X, guidedsegs, ll, tree::Tree, f, g, θ, rho)
    T = tree.T
    for i in eachindex(T)
        i == 1 && continue  # skip root-node (has no parent)
        i′ = tree.Par[i]
        llu = ll[i'] * tree.lastone[i]
        κ = MSDE.SDEKernel(f, g, messages[i].ts, θ)
        if rho == 0 || !isassigned(guidedsegs, i)
            Z = nothing
        else
            if guidedsegs[i].W isa DiffEqNoiseProcess.NoiseProcess
                Z = pCN(guidedsegs[i].W, rho) # think about modifying?
            else
                Z = pCN(guidedsegs[i].W.source, rho)
            end
        end
        solfw, llnew = MSDE.forwardguiding(κ, messages[i], (X[i′], llu), Z; inplace=false, save_noise=true)
        ll[i] = llnew
        X[i] = 𝕏(solfw[end][1:end-1])
        guidedsegs[i] = solfw
    end
    ll_leaves = sum(ll[tree.lids])
    X, guidedsegs, ll, ll_leaves
end




# Initialisation
X = zeros(𝕏, tree.n)  # values at nodes
for id in tree.lids
    X[id] = Xd[id]
end

dt0 = 0.001
rho = 0.0
Q = [i in tree.lids ? WGaussian{(:μ,:Σ,:c)}(Vector(X[i]), 0.1Matrix(I(d)), 0.0) : missing for i in eachindex(X)]
Q, messages = bwfiltertree!(Q, tree, θlin, dt0)

# init, set types
guidedsegs = Vector{Any}(undef, tree.n) # save all guided segments
X, guidedsegs, ll, ll_leaves = fwguidtree!(messages, X, guidedsegs, zeros(tree.n), tree, f, g, θ, rho)
G = typeof(guidedsegs[2])

# small test if pCN works
ll2 = zeros(tree.n)
X, guidedsegs2, ll2, ll_leaves2 = fwguidtree!(messages, X, copy(guidedsegs), ll2, tree, f, g, θ, 1.0)

@show ll-ll2 # expected: all zeros
# @show hcat(guidedsegs[3].u- guidedsegs2[3].u) # expected: all zeros

guidedsegs[2].W.W == pCN(guidedsegs[2].W, 1.0).source.W


## plotting

cols = repeat([:blue, :red, :magenta, :orange],2)
pl = scatter(tree.T, getindex.(X,1),color=:red)
scatter!(tree.T, getindex.(X,2),color=:black)
for j ∈ 1:d
    #pl = scatter(T, getindex.(X,j),color=:black)
    for i in 2:tree.n
        gg = guidedsegs[i]
        #gg = segs[i]
        #lines!(gg.t, getindex.(gg.u,j),color=cols[i])
        lines!(gg.t, getindex.(gg.u,j),color=sample(cols))
    end
    save("treecomp$j.png", pl)
end
display(pl)


## regression of drift parameters θ[1]
function driftparamstree(prior, segs, tree::Tree, f, g, θ, paramjac, messages)
    G = deepcopy(prior) # posterior that should be returned
    for i in eachindex(tree.T)
        i == 1 && continue  # skip root-node  (has no parent)
        # define timegrid on segment
        i′ = tree.Par[i]
        # t = T[i′]
        # m = round(Int, (T[i]-t)/dt0)
        # dt = (T[i] - t)/m
        #κ = MSDE.SDEKernel(f, g, t:dt:T[i], θ)
        ts = messages[i].ts
        κ = MSDE.SDEKernel(f, g, ts, θ)
        R = MSDE.Regression(κ,θ=θ[1],paramjac=paramjac)

        G = MSDE.conjugate(R, map(x->x[1:end-1], segs[i].u), G, ts)  # map because last element is the loglikelihood (this is is we provide segs as guidedsegs)
    end
    G
end
# Estimate
# Gaussian prior
prior = MS.Gaussian{(:F,:Γ)}(zeros(2), Matrix(0.1*I(2)))
#yprototype = segs[2].prob.u0 # would be needed for Jacobian computation by AD
# paramjac for non AD version
function paramjac(u,p,t)
  Diagonal(M*u)
end
G = driftparamstree(prior, guidedsegs, tree, f, g, θ, paramjac, messages)
p̂ = mean(G)
se = sqrt.(diag(cov(G)))
display(map((p̂, se, p) -> "$(round(p̂, digits=3)) ± $(round(se, digits=3)) (true: $p)", p̂, se, θ[1]))


function mcmc(θ, prior, iters, X, tree; dt0 = 0.001, rho=0.9)
    θs = [θ]
    n = tree.n
    Q = [i in tree.lids ? WGaussian{(:μ,:Σ,:c)}(Vector(X[i]), 0.01Matrix(I(d)), 0.0) : missing for i in eachindex(X)]

    guidedsegs = Vector{Any}(undef, n) # save all guided segments
    ll = 0.0
    llprop = 0.0

    θlin = (B(θ), zeros(d), Σ(θ))
    Q, messages = bwfiltertree!(Q, tree, θlin, dt0)

    X, guidedsegs, _, ll = fwguidtree!(messages, X, guidedsegs, ll, tree, f, g, θ, 0.0)
    accepted = 0

    for iter in 1:iters
        mod(iter, 10) == 0 && println(iter)

        guidedsegs2 = copy(guidedsegs)
        #
        # G = driftparamstree(prior, guidedsegs, tree, f, g, θ, paramjac, messages)
        # θ = (rand(convert(Gaussian{(:μ, :Σ)}, G)), θ[2])
        θprop = (θ[1] + 0.01*randn(size(θ[1])...), θ[2])
        Xprop, guidedsegsprop, _, llprop = fwguidtree!(messages, X, guidedsegs2, llprop, tree, f, g, θprop, rho)
        push!(θs, θ)
        if log(rand()) < llprop - ll
            ll = llprop
            θ = θprop
            guidedsegs .= guidedsegsprop
            X .= Xprop
            accepted += 1
        end
    end
    θs, accepted/iters
end
iters = 5000
θinit = ([3.3, 1.5], [0.1, 0.1])
θs, acc = mcmc(θinit, prior, iters, X, tree;dt0=0.01)

PLOT = true
if PLOT
    fig = Figure()
    for i in 1:2
        fig[1,i] = ax = Axis(fig, title="θ$i")
        lines!(ax, 0:iters, getindex.(first.(θs), i), color=:blue)
        lines!(ax, 0:iters, fill(θ[1][i], iters+1), color=:green)
    end
    fig
end
display(fig)
