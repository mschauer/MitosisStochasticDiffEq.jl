using MitosisStochasticDiffEq
using Test

println("Run tests.")
@time include("sample_test.jl")
@time include("filter_test.jl")
@time include("informationfilter_test.jl")
@time include("lyapunovfilter_test.jl")
@time include("guiding_test.jl")
@time include("static_array_test.jl")
@time include("mitosis_test.jl")
@time include("regression_test.jl")
@time include("ensemble_test.jl")
