using Pkg
path = @__DIR__
cd(path)
Pkg.activate(path)
using Mitosis
using MitosisStochasticDiffEq
using LinearAlgebra, Statistics, Random, StatsBase
using DelimitedFiles
using StaticArrays
using NewickTree
using DifferentialEquations
using DiffEqNoiseProcess
using Plots

const d = 2
const 𝕏 = SVector{d,Float64}
const 𝕏_ = SVector{d+1,Float64}
const MS = Mitosis
const MSDE = MitosisStochasticDiffEq

PLOT = true

include("tree.jl")
include("sdetree.jl")

## Read tree
tree = Tree(S50)
#print_tree(tree.newick)

## define model (two traits, as in presentation)
bθ = @SVector [.5, 0.9]
σ0 = @SVector [1.25, 1.35]
θ0 = (bθ, σ0)

const M = @SMatrix [-1.0 1.0; 1.0 -1.0]
f(u,θ,t) = 𝕏(tanh.(Diagonal(θ[1]) * M * u) )  # f(u,θ,t) = Diagonal(θ[1]) * M * u
g(u,θ,t) = θ[2]

## forward sample on the tree to simulate the observations at the tips
dt0 = 0.001
u0 = zero(𝕏)  # value at root node
Xd, segs = forwardsample(tree, u0, θ0, dt0, f, g)


## Define auxiliary process
B(θ) = (Diagonal(θ[1]) * M)
Σ(θ) = SMatrix{2,2}(Diagonal(θ[2]))
b̃ = @SVector [.1, 0.1]  #θ̃ = θ0
θ̃ = (b̃, σ0)
θlin = Array.((B(θ̃), @SVector(zeros(d)), Σ(θ̃)))


function mcmc(tree, Xd, f, g, θlin, θinit; ρ=0.99, iters=5000, dt=0.01, σprop=0.05, precisionatleaves=10e-6)
    Q = [i in tree.lids ? WGaussian{(:μ,:Σ,:c)}(Vector(Xd[i]), precisionatleaves*Matrix(I(d)), 0.0) : missing for i in tree.ids]
    Q, messages = bwfiltertree!(Q, tree, θlin, dt)

    guidedsegs = Vector{Any}(undef, tree.n) # save all guided segments
    X = zeros(𝕏, tree.n)  # values at nodes
    for id in tree.lids
        X[id] = Xd[id]
    end

    Z = [innov(messages[i].ts) for i ∈ 2:tree.n]
    θ = θinit
    X, guidedsegs, ll, 𝐋 = fwguidtree!(X, guidedsegs, messages, tree, f, g, θ, Z)


    Xᵒ, guidedsegsᵒ = deepcopy(X), deepcopy(guidedsegs)

    θs = [θinit]
    accepted = 0

    for iter in 1:iters
        θᵒ = (θ[1] + σprop * randn(size(θ[1])...), θ0[2])
        Zᵒ = [pcn_innov(Z[i], ρ) for i ∈ eachindex(Z)]
        Xᵒ, guidedsegsᵒ, llᵒ, 𝐋ᵒ = fwguidtree!(Xᵒ, guidedsegsᵒ, messages, tree, f, g, θᵒ, Zᵒ)
        Δ = 𝐋ᵒ - 𝐋

        if mod(iter, 100) == 0
             println(iter,"   ", round.(θ[1];digits=2),"   ", round.(θᵒ[1];digits=2), "       ",
                    round(𝐋;digits=3), "   ", round(𝐋ᵒ;digits=3),"   ", round(Δ; digits=3))
        end

        if log(rand()) < Δ
            𝐋 = 𝐋ᵒ
            ll .= llᵒ
            θ = θᵒ
            guidedsegs, guidedsegsᵒ = guidedsegsᵒ, guidedsegs # don't care about this
            X, Xᵒ = Xᵒ, X # don't care about this
            Z, Zᵒ = Zᵒ, Z    #  Z .= Zᵒ
            accepted += 1
        end
        push!(θs, deepcopy(θ))
    end

    θs, guidedsegs, accepted/iters
end

θinit = ([4.5, 0.1], σ0)
iters = 5_000
θs, guidedsegs, frac_accepted = mcmc(tree, Xd, f, g, θlin, θinit; iters=iters)

## summary stats
burnin = div(3iters,4):iters
θs1 = getindex.(θs,1)[burnin]
@show mean(getindex.(θs1,1))
@show mean(getindex.(θs1,2))
@show θ0[1][1]
@show θ0[1][2]

## plotting
PLOT = true
if PLOT==true
    include("plottingtree.jl")
end

## partially conjugate steps
STOP = true
if STOP==false
    ## regression of drift parameters θ[1]
    function driftparamstree(prior, segs, tree::Tree, f, g, θ, paramjac, messages)
        G = deepcopy(prior) # posterior that should be returned
        for i in eachindex(tree.T)
            i == 1 && continue  # skip root-node  (has no parent)
            ts = messages[i].ts
            κ = MSDE.SDEKernel(f, g, ts, θ)
            R = MSDE.Regression(κ,θ=θ[1],paramjac=paramjac)
            G = MSDE.conjugate(R, map(x->x[1:end-1], segs[i].u), G, ts)  # map because last element is the loglikelihood (this is is we provide segs as guidedsegs)
        end
        G
    end

    # Gaussian prior
    prior = MS.Gaussian{(:F,:Γ)}(zeros(2), Matrix(0.001*I(2)))
    # paramjac for non AD version
    function paramjac(u,p,t)
      Diagonal(M*u)
    end
    G = driftparamstree(prior, guidedsegs, tree, f, g, θ, paramjac, messages)
    p̂ = mean(G)
    se = sqrt.(diag(cov(G)))
    display(map((p̂, se, p) -> "$(round(p̂, digits=3)) ± $(round(se, digits=3)) (true: $p)", p̂, se, θ0[1]))
end
