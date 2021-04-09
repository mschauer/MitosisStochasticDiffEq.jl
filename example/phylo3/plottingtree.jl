using Plots

function plotsegment!(pl1, pl2, leafid, segs, col)
    i = leafid
    atroot = false
    while !atroot
        gg = segs[i]
        plot!(pl1, gg.t, getindex.(gg.u,1),color=col)
        plot!(pl2, gg.t, getindex.(gg.u,2),color=col)
        ipar = tree.Par[i]
        i = ipar
        if i==1
            atroot=true
        end
    end
end

## trace plots of pars
fig1 = plot(0:iters, getindex.(first.(θs), 1), color=:blue)
fig2 = plot(0:iters, getindex.(first.(θs), 2), color=:blue)
plot!(fig1, 0:iters, fill(θ0[1][1], iters+1), color=:green)
plot!(fig2, 0:iters, fill(θ0[1][2], iters+1), color=:green)
pl = plot(fig1, fig2, layout = (2, 1), legend = false)
png("trace_theta.png")



## forward simulated paths
cols = repeat([:blue, :red, :magenta, :orange],2)
pl1 = scatter(tree.T[tree.lids], getindex.(Xd,1)[tree.lids],color=:black,legend=false)
pl2 = scatter(tree.T[tree.lids], getindex.(Xd,2)[tree.lids],color=:black,legend=false)
for i in 2:tree.n
    global gg
    gg = segs[i]
    col = sample(cols)
    plot!(pl1, gg.t, getindex.(gg.u,1),color=col)
    plot!(pl2, gg.t, getindex.(gg.u,2),color=col)
end
pl = plot(pl1, pl2, layout=(2,1), legend=false)
display(pl)
png("dataforward.png")

## some segments of guided process at final iteration
leaves = Xd[tree.lids]
tleaves = tree.T[tree.lids]
pl1 = scatter(tleaves, getindex.(leaves,1) ,color=:black, legend=false)
pl2 = scatter(tleaves, getindex.(leaves,2) ,color=:black, legend=false)

cols = repeat([:blue, :red, :green, :magenta, :orange],2)
for i ∈ 0:4
    plotsegment!(pl1, pl2, tree.lids[end-i], guidedsegs, cols[i+1])
end
display(pl)
pl = plot(pl1, pl2, layout = (2, 1), legend = false)
display(pl)
png("guidedforward.png")
