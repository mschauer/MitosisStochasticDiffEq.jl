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


mutable struct Regression{kernelType,pJType,ifuncType,phiType,uType}
  k::kernelType
  fjac!::pJType
  ϕ0func!::ifuncType
  ϕ::phiType
  ϕ0::uType
  y::uType
  y2::uType
end

function Regression(sdekernel,yprototype,ϕprototype;
      paramjac=nothing,intercept=nothing)

  y = similar(yprototype)
  y2 = similar(y)
  ϕ0 = similar(y)

  ϕ = similar(ϕprototype)

  Regression{typeof(sdekernel),typeof(paramjac),typeof(intercept),typeof(ϕ),typeof(y)}(sdekernel,paramjac,intercept,ϕ,ϕ0,y,y2)
end
