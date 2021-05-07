outer_(x) = x*x'
outer_(x::Number) = x*x'*I
outer_(x::AbstractVector) = Diagonal(x.*x)

construct_a(σ::Number,P) = outer_(σ)*similar_type(P, Size(size(P,1),size(P,1)))(I)
construct_a(σ::Number,P::Number) = σ*σ'
construct_a(σ,P) = Matrix(outer_(σ))

get_dt(ts::AbstractRange) = step(ts)
get_dt(ts::AbstractVector) = ts[2] - ts[1]
get_tspan(ts) = (first(ts),last(ts))

timechange(s, tstart=first(s), tend=last(s)) = tstart .+ (s .- tstart).*(2.0 .- (s .- tstart)/(tend-tstart))

isinplace(R::AbstractRegression{inplace}) where {inplace} = inplace

# linear approximation
(a::AffineMap)(u,p,t) = a.B*u .+ a.β
(a::ConstantMap)(u,p,t) = a.x

function convert_message(message, F1::InformationFilter, F2::CovarianceFilter)
  @unpack ktilde, ts, soldis, sol = message
  # P = inv(H)

  # ν = inv(H)F

  return Message(ktilde, ts, soldis, nothing)
end

function convert_message(message, F1::CovarianceFilter, F2::InformationFilter)
  @unpack ktilde, ts, soldis, sol = message
  # H = inv(P)

  # F = Hν

  return Message(ktilde, ts, soldis, nothing)
end


# handle noise conversion between solvers
function compute_Z(::Nothing, ::Nothing, trange, u0)
    n = length(trange)

    return collect(zip(1:n, trange, cumsum([[zero(u0)];[sqrt(trange[i+1]-ti)*randn(size(u0))
            for (i,ti) in enumerate(trange[1:end-1])]])))
end

function compute_Z(::Nothing, noise_rate_prototype, trange, u0)
    n = length(trange)

    return collect(zip(1:n, trange, cumsum([[zeros(size(noise_rate_prototype,2))];[sqrt(trange[i+1]-ti)*randn(size(noise_rate_prototype,2))
            for (i,ti) in enumerate(trange[1:end-1])]])))
end

function compute_Z(Z::DiffEqNoiseProcess.NoiseGrid, noise_rate_prototype, trange, u0)
    n = length(trange)

    return collect(zip(1:n, trange, Z.W))
end

function compute_Z(Z::DiffEqNoiseProcess.NoiseWrapper, noise_rate_prototype, trange, u0)
    n = length(trange)

    return collect(zip(1:n, trange, Z.source.W))
end

function compute_Z(Z, noise_rate_prototype, trange, u0)
    n = length(trange)

    return collect(zip(1:n, trange, Z))
end
