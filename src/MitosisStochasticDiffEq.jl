module MitosisStochasticDiffEq

using Mitosis
using RecursiveArrayTools
using StochasticDiffEq
using OrdinaryDiffEq
using DiffEqNoiseProcess
using LinearAlgebra
using Random
using UnPack
using Statistics
using StaticArrays
using ForwardDiff

outer_(x) = x*x'
outer_(x::Number) = Diagonal([x*x'])
outer_(x::AbstractVector) = Diagonal(x.*x)

include("types.jl")
include("sample.jl")
include("filter.jl")
include("guiding.jl")
include("regression.jl")
include("derivative_utils.jl")

end
