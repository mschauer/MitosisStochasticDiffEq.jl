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
