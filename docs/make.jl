using MitosisStochasticDiffEq
using Documenter

makedocs(;
    modules=[MitosisStochasticDiffEq],
    authors="mschauer <moritzschauer@web.de> and contributors",
    repo="https://github.com/mschauer/MitosisStochasticDiffEq.jl/blob/{commit}{path}#L{line}",
    sitename="MitosisStochasticDiffEq.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mschauer.github.io/MitosisStochasticDiffEq.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mschauer/MitosisStochasticDiffEq.jl",
)
