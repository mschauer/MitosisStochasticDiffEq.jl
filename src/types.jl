abstract type AbstractSDEKernel
end

struct SDEKernel{fType,gType,tType,pType,NRPType} <: AbstractSDEKernel
  f::fType
  g::gType
  trange::tType
  p::pType
  noise_rate_prototype::NRPType
  constant_diffusity::Bool
end

struct SDEKernel!{fType,gType,gstep!Type,tType,pType,NRPType,wsType} <: AbstractSDEKernel
  f::fType
  g::gType
  gstep!::gstep!Type
  trange::tType
  p::pType
  noise_rate_prototype::NRPType
  ws::wsType
  constant_diffusity::Bool
end
function make_gstep(g) #g has signature  g(du,u,p,t)
  function gstep!(dx, ws, x, p, t, dw, noise_rate_prototype)   
    if noise_rate_prototype===nothing
      g(ws, x, p, t)
      dx .+= ws .* dw 
    else
      g(ws, x, p,t)
      dx .+= ws * dw 
    end
  end
end

function SDEKernel(f,g,trange,p=nothing,noise_rate_prototype=nothing,constant_diffusity=false)
  SDEKernel{typeof(f),typeof(g),typeof(trange),typeof(p),typeof(noise_rate_prototype)}(f,g,trange,p,noise_rate_prototype,constant_diffusity)
end

function SDEKernel!(f,g,gstep!,trange,p=nothing,noise_rate_prototype=nothing,constant_diffusity=false; ws = copy(noise_rate_prototype))
  SDEKernel!{typeof(f),typeof(g),typeof(gstep!),typeof(trange),typeof(p),typeof(noise_rate_prototype),typeof(ws)}(f,g,gstep!, trange,p,noise_rate_prototype, ws, constant_diffusity)
end
abstract type AbstractFilteringAlgorithm end

struct CovarianceFilter <: AbstractFilteringAlgorithm end
struct InformationFilter <: AbstractFilteringAlgorithm end
struct LyapunovFilter <: AbstractFilteringAlgorithm end

struct solLyapunov{kernelType,ΣType,νTType,PTType,CTType}
  ktilde::kernelType
  Σ::ΣType
  νT::νTType
  PT::PTType
  cT::CTType
end

struct Message{kernelType,solType,sol2Type,tType,filterType}
  ktilde::kernelType
  sol::solType
  soldis::sol2Type
  ts::tType
  filter::filterType
end

function Message(sol, sdekernel::SDEKernel, filter, apply_timechange=false)
  soldis = reverse(Array(sol), dims=2)
  if OrdinaryDiffEq.isadaptive(sol.alg) || apply_timechange
    ts = reverse(sol.t)
  else
    ts = sdekernel.trange
  end
  Message{typeof(sdekernel),typeof(sol),typeof(soldis),typeof(ts),typeof(filter)}(sdekernel,
    sol,soldis,ts,filter)
end

struct GuidingDriftCache{kernelType,messageType}
  k::kernelType
  message::messageType
end

struct GuidingDiffusionVectorCache{gType}
  g::gType
end

struct GuidingDiffusionCache{gType,sizeType}
  g::gType
  padded_size::sizeType
end

abstract type AbstractRegression{inplace} end

struct Regression{kernelType,pJType,ifuncType,pfType,θType,duType,inplace} <: AbstractRegression{inplace}
  k::kernelType
  fjac::pJType
  ϕ0func::ifuncType
  pf::pfType
  θ::θType
  dy::duType
  isscalar::Bool
end

function Regression{iip}(sdekernel::Union{SDEKernel,SDEProblem}; paramjac=nothing,
    intercept=nothing, θ=sdekernel.p, yprototype=nothing,
    dyprototype=nothing, m=nothing) where iip

  if paramjac === nothing
    if iip
      pf = ParamJacobianWrapper2(sdekernel.f,first(sdekernel.trange),yprototype)
    else
      pf = ParamJacobianWrapper(sdekernel.f,first(sdekernel.trange),yprototype)
    end
  else
    pf = nothing
  end

  if !iip
    dy = nothing
  else
    dyprototype === nothing && error("dyprototype needs to be known for using inplace functions.")
    if m===nothing
      # default, diagonal noise
      dy = similar(dyprototype)
      isscalar = false
    else
      if m==1
        # scalar noise case
        dy = similar(dyprototype, length(dyprototype))
        isscalar = true
      else
        # non-diagonal noise
        dy = similar(dyprototype, length(dyprototype,m))
        isscalar = false
      end
    end
  end

  if m===nothing
    isscalar = false
  else
    isscalar = (m==1)
  end

  Regression{typeof(sdekernel),typeof(paramjac),typeof(intercept),typeof(pf),
    typeof(θ),typeof(dy),iip}(sdekernel,
    paramjac,intercept,pf,θ,dy,isscalar)
end

function Regression(sdekernel;kwargs...)
  iip=DiffEqBase.isinplace(sdekernel.g,4)
  Regression{iip}(sdekernel;kwargs...)
end


mutable struct Regression!{kernelType,pJType,ifuncType,phiType,uType,pfType,θType,duType,inplace} <: AbstractRegression{inplace}
  k::kernelType
  fjac!::pJType
  ϕ0func!::ifuncType
  ϕ::phiType
  ϕ0::uType
  y::uType
  y2::uType
  pf::pfType
  θ::θType
  dy::duType
  isscalar::Bool
end

function Regression!{iip}(sdekernel::Union{SDEKernel,SDEProblem}, yprototype;
      paramjac_prototype=nothing, paramjac=nothing, intercept=nothing, isdiagonal=true,
      θ=sdekernel.p,
      dyprototype=yprototype, m=length(yprototype)) where iip

  y = similar(yprototype)
  y2 = similar(y)
  ϕ0 = similar(y)

  if paramjac_prototype !== nothing
    ϕ = similar(paramjac_prototype)
  else
    ϕ = zeros((length(yprototype),length(θ)))
  end

  if paramjac === nothing
    if iip
      pf = ParamJacobianWrapper2(sdekernel.f,first(sdekernel.trange),y)
    else
      pf = ParamJacobianWrapper(sdekernel.f,first(sdekernel.trange),y)
    end
  else
    pf = nothing
  end

  if !iip
    dy = nothing
  else
    dyprototype === nothing && error("dyprototype needs to be known for using inplace functions.")
    if m==length(yprototype)
      # default, diagonal noise
      dy = similar(dyprototype)
    else
      if m==1
        # scalar noise case
        dy = similar(dyprototype, length(dyprototype))
      else
        # non-diagonal noise
        dy = similar(dyprototype, length(dyprototype,m))
      end
    end
  end

  isscalar = (m==1 && length(yprototype)!=1)

  Regression!{typeof(sdekernel),typeof(paramjac),typeof(intercept),typeof(ϕ),
    typeof(y),typeof(pf),typeof(θ),typeof(dy),iip}(sdekernel,paramjac,intercept,
    ϕ,ϕ0,y,y2,pf,θ,dy,isscalar)
end

function Regression!(sdekernel,yprototype;kwargs...)
  iip=DiffEqBase.isinplace(sdekernel.g,4)
  Regression!{iip}(sdekernel,yprototype;kwargs...)
end

# internal solvers
abstract type AbstractInternalSolver end

struct EulerMaruyama! <: AbstractInternalSolver end

struct DefaultForwardGuidingP end

"""
    tangent!(du, u, dz, P)

!! May change `du`, but has to return it. !!

"""
function tangent!
end
