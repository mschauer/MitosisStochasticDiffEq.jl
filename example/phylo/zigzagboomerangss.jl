using ZigZagBoomerang
using SparseArrays
using StructArrays
using ForwardDiff
using ForwardDiff: Dual, value, partials
const D1 = Dual{Nothing, Float64, 1}
const D𝕏 = typeof(D1.(zero(𝕏)))






function sfwguidtree!(X, guidedsegs, messages, tree::Tree, f, g, θ, Z; apply_time_change=false)
    ll = zeros(D1, tree.n)
    l = rand(tree.lids)
    chance = zeros(tree.n)
    chance[tree.lids] .= 1/length(tree.lids)
    active = zeros(Bool, tree.n)
    active[l] = true
    for i in reverse(eachindex(tree.T))
        i == 1 && continue
        active[tree.Par[i]] = active[i]
        chance[tree.Par[i]] += chance[i]
    end
    @assert chance[1] ≈ 1.0
    for i in eachindex(tree.T)
        i == 1 && continue  # skip root-node (has no parent)
        ipar = tree.Par[i]
        if active[i] 
            κ = MSDE.SDEKernel(f, g, messages[i].ts, θ)
            solfw, llnew = MSDE.forwardguiding(κ, messages[i], (X[ipar], 0.0), Z[i-1]; inplace=false, save_noise=true, apply_timechange=apply_time_change)
            X[i] = D𝕏(solfw[end][1:end-1])
            ll[i] = llnew/chance[i] + ll[ipar] * tree.lastone[i]
            guidedsegs[i] = solfw
        else
            ll[i] = ll[ipar] * tree.lastone[i]
        end

    end
    𝐋 = sum(ll[tree.lids])
    X, guidedsegs, ll, 𝐋
end


θinit = [4.5, 0.1], σ0
iters = 5_000
#θs, guidedsegs, frac_accepted = mcmc(tree, Xd, f, g, θlin, θinit; iters=iters)
σprop = 0.05
precisionatleaves=1e-4
dt = 0.01

Q = [i in tree.lids ? WGaussian{(:μ,:Σ,:c)}(Vector(Xd[i]), precisionatleaves*Matrix(I(d)), 0.0) : missing for i in tree.ids]
Q, messages = bwfiltertree!(Q, tree, θlin, dt)

f(u,θ,t) = SVector((tanh.(Diagonal(θ[1]) * M * u))...)  # f(u,θ,t) = Diagonal(θ[1]) * M * u

guidedsegs = Vector{Any}(undef, tree.n) # save all guided segments
X = zeros(D𝕏, tree.n)  # values at nodes
for id in tree.lids
    X[id] = Xd[id]
end


Z = [innov(messages[i].ts) for i ∈ 2:tree.n]

F(x, Z) = sfwguidtree!(X, guidedsegs, messages, tree, f, g, (x, σ0), Z)[4]

t0 = 0.0
x0 = [1.5, 0.5]
N = length(x0)

function mk_∇ϕi(ρ)
    ith = zeros(N)
    Z = [innov(messages[i].ts) for i ∈ 2:tree.n]
    function (x,i)
        Z[:] = [pcn_innov(Z[i], ρ) for i ∈ eachindex(Z)]
        ith[i] = 1
        sa = StructArray{ForwardDiff.Dual{}}((x, ith))
        δ = F(sa, Z).partials[]
        ith[i] = 0
        return 0.1*x[i] - δ
    end
end

v0 = rand((-1.0,1.0), N)
c = 1.0
T = 300.
Γ = sparse(1.0*I(N))
tr, _, (acc, num), cs = @time ZigZagBoomerang.spdmp(mk_∇ϕi(0.99), t0, x0, v0, T, c*ones(N), ZigZag(Γ, 0*x0); adapt=true, structured=false, progress=true)


ts, θs = ZigZagBoomerang.sep(tr)
## trace plots of pars
fig1 = plot(ts, getindex.(θs, 1), color=:blue)
fig2 = plot(ts, getindex.(θs, 2), color=:blue)
plot!(fig1, ts, fill(θ0[1][1], length(ts)), color=:green)
plot!(fig2, ts, fill(θ0[1][2], length(ts)), color=:green)
pl = plot(fig1, fig2, layout = (2, 1), legend = false)
display(pl)
png("zz_trace_theta0.png")

pl