using Pkg
path = @__DIR__
cd(path)
Pkg.activate(path)

using MitosisStochasticDiffEq
using LinearAlgebra, Statistics, Random, StatsBase
using DelimitedFiles
using StaticArrays
using NewickTree
using DifferentialEquations
using DiffEqNoiseProcess


const d = 2
const 𝕏 = SVector{d,Float64}

const MSDE = MitosisStochasticDiffEq

# using MeasureTheory
# import MeasureTheory.logdensity
MSDE.dim(p::MSDE.WGaussian{(:μ,:Σ,:c)}) = length(p.μ)
MSDE._logdet(p::MSDE.WGaussian{(:μ,:Σ,:c)}) = MSDE._logdet(p.Σ, MSDE.dim(p))
MSDE.whiten(p::MSDE.WGaussian{(:μ,:Σ,:c)}, x) = MSDE.lchol(p.Σ)\(x - p.μ)
MSDE.sqmahal(p::MSDE.WGaussian, x) = MSDE.norm_sqr(MSDE.whiten(p, x))
MSDE.logdensityof(p::MSDE.WGaussian{(:μ, :Σ, :c)}, x) = p.c - (MSDE.sqmahal(p, x) + MSDE._logdet(p) + MSDE.dim(p) * log(2pi)) / 2

include("tree.jl")
include("sdetree.jl")

Random.seed!(10)

## Read tree
tree = [Tree(S50), Tree(S20coal), Tree(S)][1]
#print_tree(tree.newick)

## define model (two traits, as in presentation)
bθ = @SVector [.5, 1.2]
σ0 = @SVector [log(0.2), log(1.05)]
θ0 = (bθ, σ0)

#const M = [-1.0 1.0; 1.0 -1.0]
const M = SMatrix{d,d}(-1.0, 1.0, 1.0, -1.0)
f(u,θ,t) = 𝕏(tanh.(Diagonal(θ[1]) * M * u) )  # f(u,θ,t) = Diagonal(θ[1]) * M * u
g(u,θ,t) = Diagonal(exp.(θ[2]))

## forward sample on the tree to simulate the observations at the tips
dt0 = 0.001
u0 = zero(𝕏)  # value at root node
Xd, segs = forwardsample(tree, u0, θ0, dt0, f, g, EM(false))



## Define auxiliary process
B(θ) = Diagonal(θ[1]) * M
##Σ(θ) = Diagonal(exp.(θ[2]))
#σ̃(θ) = SMatrix{d,d}(exp(θ[2][1]), 0.0, 0.0, exp(θ[2][2])) # or?  g(0,θ,0) * g(0, θ, 0)'
σ̃(θ) = g(0,θ,0)

function mcmc2(tree, Xd, f, g, θinit, prior;
                 ρ=0.99,
                 iters=5000,
                 dt=0.01,
                 σprop=0.05,
                 precisionatleaves=10e5,
                 apply_time_change=true,
                 𝒫=(:μ,:Σ,:c),  # 𝒫=(:F,:\Gamma,:c)
                 recomputeguidingterm=true,
                 alg=Tsit5(),
                 SDEalg=EM(false)
                 )
    #σ0 = θinit[2]

    if 𝒫==(:μ,:Σ,:c)
        #Q = [i in tree.lids ? MSDE.WGaussian{(:μ,:Σ,:c)}(Vector(Xd[i]), precisionatleaves*Matrix(I(d)), 0.0) : missing for i in tree.ids]
        # in case of static arrays
        leavescov = inv(precisionatleaves) * SA_F64[1 0; 0 1]
        Q = [i in tree.lids ? MSDE.WGaussian{(:μ,:Σ,:c)}(Xd[i], leavescov, 0.0) : missing for i in tree.ids]
    elseif 𝒫==(:F,:Γ,:c)
        leavesprecision = precisionatleaves * SA_F64[1 0; 0 1]
        Q = [i in tree.lids ? MSDE.WGaussian{(:F,:Γ,:c)}(leavesprecision*Xd[i], leavesprecision, 0.0) : missing for i in tree.ids]
    else
        @error "𝒫 not defined"
    end
    θ = θinit
    θlin = (B(θinit), zeros(d), σ̃(θinit))

    Qtest = deepcopy(Q)

    Q, messages = bwfiltertree!(Q, tree, θlin, dt; apply_time_change=apply_time_change, alg=alg)
    Qᵒ, messagesᵒ = Q, messages

    guidedsegs = Vector{Any}(undef, tree.n) # save all guided segments
    X = zeros(𝕏, tree.n)  # values at nodes
    for id in tree.lids
        X[id] = Xd[id]
    end

    # for i in tree.lids
    #     @show Qtest[i] == Qᵒ[i]  Qtest[i] == Q[i]  Q[i] == Qᵒ[i]
    # end
    Z = [myinnov(messages[i].ts, 𝕏) for i ∈ 2:tree.n]

    X, guidedsegs, ll, 𝐋 = fwguidtree!(X, guidedsegs, Q, messages, tree, f, g, θ, Z, SDEalg)

    Xᵒ, guidedsegsᵒ, Qᵒ = deepcopy(X), deepcopy(guidedsegs), deepcopy(Q)
    θs = [θinit]
    accepted = 0

    for iter in 1:iters
        #θᵒ = (θ[1] + σprop * randn(size(θ[1])...), σ0)
        θᵒ = (θ[1] + σprop * randn(size(θ[1])...), θ[2] + σprop * randn(size(θ[2])...))
        Zᵒ =  [pcn_innov(Z[i], ρ, 𝕏) for i ∈ eachindex(Z)]
        θlinᵒ = (0.9B(θᵒ), zeros(d), σ̃(θᵒ))
        if recomputeguidingterm==true
            Qᵒ, messagesᵒ = bwfiltertree!(Qᵒ, tree, θlinᵒ, dt, apply_time_change=apply_time_change,  alg=alg)
        else
            Qᵒ = Q
        end
        Xᵒ, guidedsegsᵒ, llᵒ, 𝐋ᵒ = fwguidtree!(Xᵒ, guidedsegsᵒ, Qᵒ, messagesᵒ, tree, f, g, θᵒ, Zᵒ, SDEalg)
        Δ = 𝐋ᵒ - 𝐋 + MSDE.logdensityof(prior[1], θᵒ[1]) - MSDE.logdensityof(prior[1], θ[1]) + MSDE.logdensityof(prior[2], θᵒ[2]) - MSDE.logdensityof(prior[2], θ[2])

        if mod(iter, 10) == 0
             println(iter,"   ", round.(θ[1];digits=2),"   ", round.(θᵒ[1];digits=2), "       ",
                    round(𝐋;digits=3), "   ", round(𝐋ᵒ;digits=3),"   ", round(Δ; digits=3))
        end

        if log(rand()) < Δ
            𝐋 = 𝐋ᵒ
            ll .= llᵒ
            θ = θᵒ
            θlin = θlinᵒ
            guidedsegs, guidedsegsᵒ = guidedsegsᵒ, guidedsegs # don't care about this
            X, Xᵒ = Xᵒ, X # don't care about this
            Z, Zᵒ = Zᵒ, Z    #  Z .= Zᵒ
            Q, Qᵒ = Qᵒ, Q
            accepted += 1
        end
        push!(θs, deepcopy(θ))
    end

    θs, guidedsegs, accepted/iters, (Xᵒ, guidedsegsᵒ, Qᵒ, messagesᵒ, tree)
end




prior = (MSDE.Gaussian{(:F,:Γ)}(zeros(2), Matrix(0.01*I(2))) , MSDE.Gaussian{(:F,:Γ)}(-ones(2), Matrix(0.01*I(2))))
#θinit = (SVector(3.2, 1.0), σ0)
θinit = (SVector(3.2, 1.0), SVector(.1, .1))



iters = 50_000 # 50_000
@time θs, guidedsegs, frac_accepted, forwardguiding_input = mcmc2(tree, Xd, f, g, θinit, prior;
  iters=iters)#, 𝒫=(:F,:Γ,:c))#, dt = dt0)

begin
    trace_theta_drift = getindex.(θs, 1)
    plot([θ[1] for θ in trace_theta_drift])
    plot!([θ[2] for θ in trace_theta_drift])
    hline!([θ0[1][1]])
    hline!([θ0[1][2]])
end

begin
    trace_theta_diff = getindex.(θs, 2)
    plot([θ[1] for θ in trace_theta_diff])
    plot!([θ[2] for θ in trace_theta_diff])
    hline!([θ0[2][1]])
    hline!([θ0[2][2]])
end


## summary stats

println("fraction accepted equals: $frac_accepted")
burnin = div(3iters,4):iters
θs1 = getindex.(θs,1)[burnin]
@show mean(getindex.(θs1,1))
@show mean(getindex.(θs1,2))
@show θ0[1][1]
@show θ0[1][2]
println("-------")
θs2 = getindex.(θs,2)[burnin]
@show mean(getindex.(θs2,1))
@show mean(getindex.(θs2,2))
@show θ0[2][1]
@show θ0[2][2]

## plotting
PLOT = true
if PLOT==true
    include("plottingtree.jl")
end

# ## partially conjugate steps
# STOP = true
# if STOP==false
#     ## regression of drift parameters θ[1]
#     function driftparamstree(prior, segs, tree::Tree, f, g, θ, paramjac, messages)
#         G = deepcopy(prior) # posterior that should be returned
#         for i in eachindex(tree.T)
#             i == 1 && continue  # skip root-node  (has no parent)
#             ts = messages[i].ts
#             κ = MSDE.SDEKernel(f, g, ts, θ)
#             R = MSDE.Regression(κ,θ=θ[1],paramjac=paramjac)
#             G = MSDE.conjugate(R, map(x->x[1:end-1], segs[i].u), G, ts)  # map because last element is the loglikelihood (this is is we provide segs as guidedsegs)
#         end
#         G
#     end

#     # Gaussian prior
#     prior = MSDE.Gaussian{(:F,:Γ)}(zeros(2), Matrix(0.001*I(2)))
#     # paramjac for non AD version
#     function paramjac(u,p,t)
#       Diagonal(M*u)
#     end
#     G = driftparamstree(prior, guidedsegs, tree, f, g, θ, paramjac, messages)
#     p̂ = mean(G)
#     se = sqrt.(diag(cov(G)))
#     display(map((p̂, se, p) -> "$(round(p̂, digits=3)) ± $(round(se, digits=3)) (true: $p)", p̂, se, θ0[1]))
# end


# function mcmc(tree, Xd, f, g, θlin, θinit, prior; ρ=0.99, iters=5000, dt=0.01, σprop=0.05, precisionatleaves=10e-6)
#     σ0 = θinit[2]
#     Q = [i in tree.lids ? MSDE.WGaussian{(:μ,:Σ,:c)}(Vector(Xd[i]), precisionatleaves*Matrix(I(d)), 0.0) : missing for i in tree.ids]
#     Q, messages = bwfiltertree!(Q, tree, θlin, dt)
#
#     guidedsegs = Vector{Any}(undef, tree.n) # save all guided segments
#     X = zeros(𝕏, tree.n)  # values at nodes
#     for id in tree.lids
#         X[id] = Xd[id]
#     end
#
#     Z = [innov(messages[i].ts) for i ∈ 2:tree.n]
#     θ = θinit
#     X, guidedsegs, ll, 𝐋 = fwguidtree!(X, guidedsegs, Q, messages, tree, f, g, θ, Z)
#
#     Xᵒ, guidedsegsᵒ = deepcopy(X), deepcopy(guidedsegs)
#     θs = [θinit]
#     accepted = 0
#
#     for iter in 1:iters
#         θᵒ = (θ[1] + σprop * randn(size(θ[1])...), σ0)
#         Zᵒ = [pcn_innov(Z[i], ρ) for i ∈ eachindex(Z)]
#         Xᵒ, guidedsegsᵒ, llᵒ, 𝐋ᵒ = fwguidtree!(Xᵒ, guidedsegsᵒ, Q, messages, tree, f, g, θᵒ, Zᵒ)
#         Δ = 𝐋ᵒ - 𝐋 + logdensity(prior, θᵒ[1]) - logdensity(prior, θ[1])
#
#         if mod(iter, 100) == 0
#              println(iter,"   ", round.(θ[1];digits=2),"   ", round.(θᵒ[1];digits=2), "       ",
#                     round(𝐋;digits=3), "   ", round(𝐋ᵒ;digits=3),"   ", round(Δ; digits=3))
#         end
#
#         if log(rand()) < Δ
#             𝐋 = 𝐋ᵒ
#             ll .= llᵒ
#             θ = θᵒ
#             guidedsegs, guidedsegsᵒ = guidedsegsᵒ, guidedsegs # don't care about this
#             X, Xᵒ = Xᵒ, X # don't care about this
#             Z, Zᵒ = Zᵒ, Z    #  Z .= Zᵒ
#             accepted += 1
#         end
#         push!(θs, deepcopy(θ))
#     end
#
#     θs, guidedsegs, accepted/iters
# end
