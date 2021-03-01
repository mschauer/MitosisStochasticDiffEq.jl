mutable struct ParamJacobianWrapper{fType,tType,uType} <: Function
  f::fType
  t::tType
  u::uType
end

(ff::ParamJacobianWrapper)(p) = ff.f(ff.u,p,ff.t)

mutable struct ParamJacobianWrapper2{fType,tType,uType} <: Function
  f::fType
  t::tType
  u::uType
end

function (ff::ParamJacobianWrapper2)(p)
  du1 = similar(p, size(ff.u))
  ff.f(du1,ff.u,p,ff.t)
  return du1
end

function calc_J!(ϕ, r::Regression!, p, t)
  @unpack fjac!, y = r

  if fjac! !== nothing
    fjac!(ϕ, y, p, t)
  else
    @unpack pf = r
    pf.t = t
    pf.u = y
    ForwardDiff.jacobian!(ϕ, pf, p)
  end

  return nothing
end

function calc_J(y, r::Regression, p, t)
  @unpack fjac = r

  if fjac !== nothing
    ϕ = fjac(y, p, t)
  else
    @unpack pf = r
    pf.t = t
    pf.u = y
    ϕ = ForwardDiff.jacobian(pf, p)
  end

  return ϕ
end
