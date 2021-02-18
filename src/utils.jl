outer_(x) = x*x'
outer_(x::Number) = Diagonal([x*x'])
outer_(x::AbstractVector) = Diagonal(x.*x)

get_dt(ts::AbstractRange) = step(ts)
get_dt(ts::AbstractVector) = ts[2] - ts[1]
get_tspan(ts) = (first(ts),last(ts))

timechange(s, tstart=first(s), tend=last(s)) = tstart .+ (s .- tstart).*(2.0 .- (s .- tstart)/(tend-tstart))
