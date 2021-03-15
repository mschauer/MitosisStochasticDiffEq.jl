function conjugate(r::Union{Regression,Regression!}, Y, Γ0::AbstractArray, ts)
  Γ0_ = Matrix(Γ0)
  conjugate(r, Y, Mitosis.Gaussian{(:F,:Γ)}(zeros(eltype(Γ0_), size(Γ0_, 1)), Γ0_), ts)
end

function conjugate(r::Union{Regression,Regression!}, Y::RODESolution, Γ0::AbstractArray)
  Γ0_ = Matrix(Γ0)
  Y_ = Y.u
  ts = Y.t
  conjugate(r, Y_, Mitosis.Gaussian{(:F,:Γ)}(zeros(eltype(Γ0_), size(Γ0_, 1)), Γ0_), ts)
end

function conjugate(r::Regression, Y, prior::Gaussian, ts)
  @unpack k, fjac, ϕ0func, θ, dy, isscalar = r
  @unpack g, p = k

  t = ts[1]
  y = Y[1]

  ϕ = calc_J(y, r, θ, t)
  μ = copy(prior.F)
  Γ = copy(prior.Γ)

  for i in 1:length(Y)-1
    ϕ = calc_J(y, r, θ, t)
    if isinplace(r)
      g(dy, y, p, t)
    else
      dy = g(y, p, t)
    end
    if !isscalar
      Gϕ = pinv(outer_(dy))*ϕ
    else
      Gϕ = pinv(outer_(reshape(dy, :, 1)))*ϕ
    end
    zi = ϕ'*Gϕ
    t2 = ts[i + 1]
    y2 = Y[i + 1]

    ds = t2 - t

    # compute intercept
    if ϕ0func !== nothing
      ϕ0 = ϕ0func(y, p, t)
      μ = μ .+ Gϕ'*((y2 - y) .- ϕ0*ds)
    else
      μ = μ .+ Gϕ'*(y2 - y)
    end

    t = t2
    y = y2
    Γ = Γ .+ zi*ds

  end
  return Mitosis.Gaussian{(:F,:Γ)}(μ, Γ)
end

function conjugate(r::Regression!, Y, prior::Gaussian, ts)
  @unpack k, fjac!, ϕ0func!, ϕ, ϕ0, y, y2, θ, dy, isscalar = r
  @unpack g, p = k

  t = ts[1]
  copyto!(y, Y[1])

  calc_J!(ϕ, r, θ, t)
  μ = prior.F
  Γ = prior.Γ

  for i in 1:length(Y)-1
    calc_J!(ϕ, r, θ, t)
    if isinplace(r)
      g(dy, y, p, t)
    else
      dy = g(y, p, t)
    end
    if !isscalar
      Gϕ = pinv(outer_(dy))*ϕ
    else
      Gϕ = pinv(outer_(reshape(dy, :, 1)))*ϕ
    end
    zi = ϕ'*Gϕ
    t2 = ts[i + 1]
    copyto!(y2, Y[i + 1])

    ds = t2 - t

    # compute intercept
    if ϕ0func! !== nothing
      ϕ0func!(ϕ0, y, p, t)
      μ .= μ + Gϕ'*((y2 - y) - ϕ0*ds)
    else
      μ .= μ + Gϕ'*(y2 - y)
    end

    t = t2
    copyto!(y, y2)
    Γ .= Γ + zi*ds

  end
  return Mitosis.Gaussian{(:F,:Γ)}(μ, Γ)
end
