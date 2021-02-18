mutable struct ParamJacobianWrapper{fType,tType,uType} <: Function
  f::fType
  t::tType
  u::uType
end

(ff::ParamJacobianWrapper)(p) = ff.f(ff.u,p,ff.t)

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
