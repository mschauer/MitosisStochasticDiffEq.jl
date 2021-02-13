function conjugate(r::Regression, Y, Γ0::AbstractArray)
    Γ0_ = Matrix(Γ0)
    conjugate(r, Y, Mitosis.Gaussian{(:F,:Γ)}(similar(Γ0_, size(Γ0_, 1)), Γ0_))
end

function conjugate(r::Regression, Y, prior::Gaussian)
    @unpack k, fjac!, ϕ0func!, ϕ, ϕ0, y, y2, θ = r
    @unpack g, pest = k

    t = Y.t[1]
    y .= Y.u[1]

    calc_J!(ϕ, r, θ, t)
    μ = prior.F
    Γ = prior.Γ

    for i in 1:length(Y)-1
        calc_J!(ϕ, r, θ, t)
        Gϕ = pinv(outer_(g(y, pest, t)))*ϕ
        zi = ϕ'*Gϕ
        t2 = Y.t[i + 1]
        y2 .= Y.u[i + 1]

        ds = t2 - t

        # compute intercept
        ϕ0func!(ϕ0, y, pest, t)

        μ .= μ + Gϕ'*((y2 - y) - ϕ0*ds)
        t = t2
        y .= y2
        Γ .= Γ + zi*ds
    end
    return Mitosis.Gaussian{(:F,:Γ)}(μ, Γ)
end
