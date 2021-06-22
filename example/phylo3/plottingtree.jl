using Plots

function plotsegment!(pl1, pl2, leafid, segs, col)
    i = leafid
    atroot = false
    while !atroot
        gg =segs[i]
        t = getindex.(gg,2)
        u = getindex.(gg,3)
        plot!(pl1, t, getindex.(u,1),color=col)
        plot!(pl2, t, getindex.(u,2),color=col)
        ipar = tree.Par[i]
        i = ipar
        if i==1
            atroot=true
        end
    end
end

## trace plots of pars
fig1 = plot(0:iters, getindex.(first.(θs), 1), color=:blue, title="θ₁")
fig2 = plot(0:iters, getindex.(first.(θs), 2), color=:blue, title="θ₂")
plot!(fig1, 0:iters, fill(θ0[1][1], iters+1), color=:green)
plot!(fig2, 0:iters, fill(θ0[1][2], iters+1), color=:green)
pl = plot(fig1, fig2, layout = (2, 1), legend = false)
png("./figs/trace_theta.png")


## trace plots of pars
fig1 = plot(0:iters, getindex.(last.(θs), 1), color=:blue, title="σ₁")
fig2 = plot(0:iters, getindex.(last.(θs), 2), color=:blue, title="σ₂")
plot!(fig1, 0:iters, fill(θ0[2][1], iters+1), color=:green)
plot!(fig2, 0:iters, fill(θ0[2][2], iters+1), color=:green)
pl = plot(fig1, fig2, layout = (2, 1), legend = false)
png("./figs/trace_sigma.png")



## forward simulated paths
cols = repeat([:blue, :red, :magenta, :orange],2)
pl1 = scatter(tree.T[tree.lids], getindex.(Xd,1)[tree.lids],color=:black,legend=false, title="x₁")
pl2 = scatter(tree.T[tree.lids], getindex.(Xd,2)[tree.lids],color=:black,legend=false, title="x₂")
for i in 2:tree.n
    global gg
    gg = segs[i]
    t = getindex.(gg,2)
    u = getindex.(gg,3)
    col = sample(cols)
    plot!(pl1, t, getindex.(u,1),color=col)
    plot!(pl2, t, getindex.(u,2),color=col)
end
pl = plot(pl1, pl2, layout=(2,1), legend=false)
display(pl)
png("./figs/dataforward.png")

## just the observations
cols = repeat([:blue, :red, :magenta, :orange],2)
pl1 = scatter(tree.T[tree.lids], getindex.(Xd,1)[tree.lids],color=:black,legend=false, title="x₁")
pl2 = scatter(tree.T[tree.lids], getindex.(Xd,2)[tree.lids],color=:black,legend=false, title="x₂")
for i in 2:tree.n
    global gg
    gg = segs[i]
    t = getindex.(gg,2)
    u = getindex.(gg,3)
    col = :white
    plot!(pl1, t, getindex.(u,1),color=col)
    plot!(pl2, t, getindex.(u,2),color=col)
end
pl = plot(pl1, pl2, layout=(2,1), legend=false)
display(pl)
png("./figs/observations.png")


## some segments of guided process at final iteration
leaves = Xd[tree.lids]
tleaves = tree.T[tree.lids]
pl1 = scatter(tleaves, getindex.(leaves,1) ,color=:black, legend=false, title="x₁")
pl2 = scatter(tleaves, getindex.(leaves,2) ,color=:black, legend=false, title="x₂")

cols = repeat([:blue, :red, :green, :magenta, :orange],2)
for i ∈ 0:3
    plotsegment!(pl1, pl2, tree.lids[end-i], guidedsegs, cols[i+1])
end
display(pl)
pl = plot(pl1, pl2, layout = (2, 1), legend = false)
display(pl)
png("./figs/guidedforward.png")
