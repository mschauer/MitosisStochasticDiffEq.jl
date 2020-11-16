function conjugate(r::Regression, Y, Ξ)
  @unpack k, fjac!, ϕ0func!, ϕ, μ, Γ, ϕ0, y, y2 = r
  @unpack g, pest = k

  t = Y.t[1]
  y .= Y.u[1]

  fjac!(ϕ, y, pest, t)
  fill!(μ, zero(eltype(ϕ)))
  fill!(Γ, zero(eltype(ϕ)))

  for i in 1:length(Y)-1
    fjac!(ϕ, y, pest, t)
    Gϕ = pinv(g(y, pest, t)*g(y, pest, t)')*ϕ
    zi = ϕ'*Gϕ
    t2 = Y.t[i + 1]
    y2 .= Y.u[i + 1]

    ds = t2 - t

    # compute intercept
    ϕ0func!(ϕ0,y,pest,t)

    μ .= μ + Gϕ'*((y2 - y) - ϕ0*ds)
    t = t2
    y .= y2
    Γ .= Γ + zi*ds
  end
  WW = Γ + Ξ
  WL = (cholesky(Hermitian(WW)).U)'
  th° = WL'\(randn(size(μ))+WL\μ)
  return th°
end
