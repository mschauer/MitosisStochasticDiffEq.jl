module MitosisStochasticDiffEq

using Mitosis
using RecursiveArrayTools
using StochasticDiffEq
using OrdinaryDiffEq
using DiffEqCallbacks
using DiffEqNoiseProcess
using LinearAlgebra
using Random
#using Parameters
using UnPack
using Statistics
using StaticArrays

println("runs")
outer_(x) = x*x'
outer_(x::AbstractVector) = Diagonal(x.*x)

include("types.jl")
include("sample.jl")
include("filter.jl")
include("guiding.jl")
include("regression.jl")

end
