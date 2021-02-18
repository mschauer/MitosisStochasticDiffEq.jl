struct SDEKernel{fType,gType,tType,pType}
  f::fType
  g::gType
  trange::tType
  p::pType
end

struct Message{kernelType,solType,sol2Type,tType}
  ktilde::kernelType
  sol::solType
  soldis::sol2Type
  ts::tType
end

function Message(sol, sdekernel::SDEKernel)
  soldis = reverse(Array(sol), dims=2)
  ts = reverse(sol.t)
  Message{typeof(sdekernel),typeof(sol),typeof(soldis),typeof(ts)}(sdekernel,sol,soldis,ts)
end

struct GuidingDriftCache{kernelType,messageType}
  k::kernelType
  message::messageType
end

struct GuidingDiffusionCache{gType}
  g::gType
end


struct Regression{kernelType,pJType,ifuncType,pfType,θType}
  k::kernelType
  fjac::pJType
  ϕ0func::ifuncType
  pf::pfType
  θ::θType
end

function Regression(sdekernel::SDEKernel; paramjac=nothing,intercept=nothing,
    θ=sdekernel.p, yprototype=nothing)

  if paramjac === nothing
    pf = ParamJacobianWrapper(sdekernel.f,first(sdekernel.trange),yprototype)
  else
    pf = nothing
  end

  Regression{typeof(sdekernel),typeof(paramjac),typeof(intercept),typeof(pf),typeof(θ)}(sdekernel,
    paramjac,intercept,pf,θ)
end


mutable struct Regression!{kernelType,pJType,ifuncType,phiType,uType,pfType,θType}
  k::kernelType
  fjac!::pJType
  ϕ0func!::ifuncType
  ϕ::phiType
  ϕ0::uType
  y::uType
  y2::uType
  pf::pfType
  θ::θType
end

function Regression!(sdekernel::SDEKernel, yprototype;
      paramjac_prototype=nothing, paramjac=nothing, intercept=nothing,
      θ=sdekernel.p)

  y = similar(yprototype)
  y2 = similar(y)
  ϕ0 = similar(y)

  if paramjac_prototype !== nothing
    ϕ = similar(paramjac_prototype)
  else
    ϕ = zeros((length(yprototype),length(θ)))
  end

  if paramjac === nothing
    pf = ParamJacobianWrapper(sdekernel.f,first(sdekernel.trange),y)
  else
    pf = nothing
  end

  Regression!{typeof(sdekernel),typeof(paramjac),typeof(intercept),typeof(ϕ),
    typeof(y),typeof(pf),typeof(θ)}(sdekernel,paramjac,intercept,ϕ,ϕ0,y,y2,pf,θ)
end
