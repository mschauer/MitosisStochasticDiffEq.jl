using Plots

function plotsegment!(pl1, pl2, leafid, segs, col)
    i = leafid
    atroot = false
    while !atroot
        gg = segs[i]
        t = gg[1]
        u = gg[2]
        plot!(pl1, t, getindex.(u,1),color=col)
        plot!(pl2, t, getindex.(u,2),color=col)
        ipar = tree.Par[i]
        i = ipar
        if i==1
            atroot=true
        end
    end
end


function plotmessage!(pl1, pl2, leafid, messages, col)
    i = leafid
    atroot = false
    while !atroot
        m = messages[i]
        t = m.ts
        u = [u[1] for u in m.soldis] # mean value
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
fig1 = plot(0:iters, getindex.(first.(θs), 1), color=:blue,  title="θ₁")
fig2 = plot(0:iters, getindex.(first.(θs), 2), color=:blue, xlabel="iterations", title="θ₂")
plot!(fig1, 0:iters, fill(θ0[1][1], iters+1), lw=2.0, color=:green)
plot!(fig2, 0:iters, fill(θ0[1][2], iters+1), lw=2.0, color=:green)
pl = plot(fig1, fig2, layout = (2, 1), legend = false)
savefig(pl, "./Figures/trace_theta.png")


## trace plots of pars
fig1 = plot(0:iters, getindex.(last.(θs), 1), color=:blue, title="σ₁")
fig2 = plot(0:iters, getindex.(last.(θs), 2), color=:blue, xlabel="iterations", title="σ₂")
plot!(fig1, 0:iters, fill(θ0[2][1], iters+1), lw=2.0, color=:green)
plot!(fig2, 0:iters, fill(θ0[2][2], iters+1), lw=2.0, color=:green)
pl = plot(fig1, fig2, layout = (2, 1), legend = false)
savefig(pl, "./Figures/trace_sigma.png")



## forward simulated paths
cols = repeat([:blue, :red, :magenta, :orange],2)
pl1 = scatter(tree.T[tree.lids], getindex.(Xd,1)[tree.lids],color=:black,legend=false, title="x₁")
pl2 = scatter(tree.T[tree.lids], getindex.(Xd,2)[tree.lids],color=:black,legend=false, xlabel="time", title="x₂")
for i in 2:tree.n
    global gg
    gg = segs[i]
    t = gg[1]
    u = gg[2]
    col = sample(cols)
    plot!(pl1, t, getindex.(u,1),color=col)
    plot!(pl2, t, getindex.(u,2),color=col)
end
pl = plot(pl1, pl2, layout=(2,1), legend=false)
display(pl)
savefig(pl, "./Figures/dataforward.png")

## just the observations
cols = repeat([:blue, :red, :magenta, :orange],2)
pl1 = scatter(tree.T[tree.lids], getindex.(Xd,1)[tree.lids],color=:black,legend=false, title="x₁")
pl2 = scatter(tree.T[tree.lids], getindex.(Xd,2)[tree.lids],color=:black,legend=false,  xlabel="time",  title="x₂")
for i in 2:tree.n
    global gg
    gg = segs[i]
    t = gg[1]
    u = gg[2]
    col = :white
    plot!(pl1, t, 0*t,color=col)
    plot!(pl2, t, 0*t,color=col)
end
pl = plot(pl1, pl2, layout=(2,1), legend=false)
display(pl)
savefig(pl, "./Figures/observations.png")


## some segments of guided process at final iteration
leaves = Xd[tree.lids]
tleaves = tree.T[tree.lids]
pl1 = scatter(tleaves, getindex.(leaves,1) ,color=:black, legend=false, title="x₁")
pl2 = scatter(tleaves, getindex.(leaves,2) ,color=:black, legend=false,
 xlabel="time",  title="x₂")

cols = repeat([:blue, :red, :green, :magenta, :orange],2)
for i ∈ 0:3
    plotsegment!(pl1, pl2, tree.lids[end-i], guidedsegs, cols[i+1])
end
pl = plot(pl1, pl2, layout = (2, 1), legend = false)
display(pl)
savefig(pl, "./Figures/guidedforward.png")


## plot expected tree at final iteration
leaves = Xd[tree.lids]
tleaves = tree.T[tree.lids]
pl1 = scatter(tleaves, getindex.(leaves,1) ,color=:black, legend=false, title="x₁");
pl2 = scatter(tleaves, getindex.(leaves,2) ,color=:black, legend=false, title="x₂",  xlabel="time");

θpred = ([mean(getindex.(θs1,1)), mean(getindex.(θs1,2))], [mean(getindex.(θs2,1)), mean(getindex.(θs2,2))] )
Z = [myinnov(forwardguiding_input[4][i].ts, 𝕏, 0) for i ∈ 2:tree.n]
Xᵒ, guidedsegsᵒ, llᵒ, 𝐋ᵒ = fwguidtree!(forwardguiding_input[1],
                                     forwardguiding_input[2],
                                     forwardguiding_input[3],
                                     forwardguiding_input[4],
                                     tree, f, g, θpred, Z, EM(false); apply_time_change=false)


for i ∈ 0:3
  plotsegment!(pl1, pl2, tree.lids[end-i], guidedsegsᵒ, cols[i+1])
end
pl = plot(pl1, pl2, layout = (2, 1), legend = false)
display(pl)
savefig(pl, "./Figures/expected_guidedforward.png")

# plot messages
# cols = repeat([:blue, :red, :green, :magenta, :orange],2)
# for i ∈ 0:3
#     plotmessage!(pl1, pl2, tree.lids[end-i], messages, cols[i+1])
# end
# pl = plot(pl1, pl2, layout = (2, 1), legend = false)
# display(pl)
# savefig(pl, "./Figures/messages.png")
