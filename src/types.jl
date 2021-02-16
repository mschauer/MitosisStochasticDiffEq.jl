struct SDEKernel{fType,gType,tType,dtType,paramType1,paramType2}
  f::fType
  g::gType
  tstart::tType
  tend::tType
  dt::dtType
  pest::paramType1
  plin::paramType2
end

function SDEKernel(f,g,tstart,tend,pest,plin;dt=nothing)
  SDEKernel{typeof(f),typeof(g),typeof(tstart),
            typeof(dt),typeof(pest),typeof(plin)}(f,g,tstart,tend,dt,pest,plin)
end


struct GuidingDriftCache{kernelType,sdType,sType,tsType}
  k::kernelType
  soldis::sdType
  sol::sType
  ts::tsType
end

struct GuidingDiffusionCache{gType}
  g::gType
end


mutable struct Regression{kernelType,pJType,ifuncType,phiType,uType,pfType,θType}
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

function Regression(sdekernel::SDEKernel, yprototype;
      paramjac_prototype=nothing,
      paramjac=nothing,intercept=nothing,θ=sdekernel.pest)


  y = similar(yprototype)
  y2 = similar(y)
  ϕ0 = similar(y)

  if paramjac_prototype !== nothing
    ϕ = similar(paramjac_prototype)
  else
    ϕ = zeros((length(yprototype),length(θ)))
  end


  if paramjac === nothing
    pf = ParamJacobianWrapper(sdekernel.f,sdekernel.tstart,y)
  else
    pf = nothing
  end

  Regression{typeof(sdekernel),typeof(paramjac),typeof(intercept),typeof(ϕ),
    typeof(y),typeof(pf),typeof(θ)}(sdekernel,paramjac,intercept,ϕ,ϕ0,y,y2,pf,θ)
end
