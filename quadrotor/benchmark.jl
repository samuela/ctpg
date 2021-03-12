"""A self-contained script to test the performance of the forward and adjoint
DifferentialEquations.jl solves with the quadrotor."""

import DifferentialEquations: Tsit5, VCABM, BS3, Euler, Vern7, VCAB3
import DiffEqFlux: FastChain, FastDense, initial_params, ODEProblem, solve
import Random: seed!
import DiffEqSensitivity:
    InterpolatingAdjoint, BacksolveAdjoint, ODEAdjointProblem, ReverseDiffVJP
using BenchmarkTools

seed!(123)

module QuadrotorEnv

import StaticArrays: SA

function env(floatT, gravity, mass, Ix, Iy, Iz)
    twopi = convert(floatT, 2π)

    function dynamics(state, u)
        # See Eq 2.25.
        x, y, z, ψ, θ, ϕ, ẋ, ẏ, ż, p, q, r = state
        sinψ, cosψ = sincos(ψ)
        sinθ, cosθ = sincos(θ)
        sinϕ, cosϕ = sincos(ϕ)
        tanθ = sinθ / cosθ

        g_1_7 = -1 / mass * (sinϕ * sinψ + cosϕ * cosψ * sinθ)
        g_1_8 = -1 / mass * (cosψ * sinϕ - cosϕ * sinψ * sinθ)
        g_1_9 = -1 / mass * cosϕ * cosθ
        SA[
            ẋ,
            ẏ,
            ż,
            q*sinϕ/cosθ+r*cosϕ/cosθ,
            q*cosϕ-r*sinϕ,
            p+q * sinϕ * tanθ+r * cosϕ * tanθ,
            g_1_7*u[1],
            g_1_8*u[1],
            gravity+g_1_9*u[1],
            (Iy - Iz) / Ix*q*r+u[2]/Ix,
            (Iz - Ix) / Iy*p*r+u[3]/Iy,
            (Ix - Iy) / Iz*p*q+u[4]/Iz,
        ]
    end

    function cost(state, u)
        x, y, z, ψ, θ, ϕ, ẋ, ẏ, ż, p, q, r = state
        (x^2 + y^2 + z^2) + (ẋ^2 + ẏ^2 + ż^2) + 0.01 * u' * u
    end

    function sample_x0()
        x = rand() * 5 - 2.5
        y = rand() * 5 - 2.5
        z = rand() * 5 - 2.5
        ψ = randn() * 0.1
        θ = randn() * 0.1
        ϕ = randn() * 0.1
        ẋ = 0
        ẏ = 0
        ż = 0
        p = randn() * 0.1
        q = randn() * 0.1
        r = randn() * 0.1
        convert(Array{floatT}, [x, y, z, ψ, θ, ϕ, ẋ, ẏ, ż, p, q, r])
    end

    function observation(state)
        x, y, z, ψ, θ, ϕ, ẋ, ẏ, ż, p, q, r = state
        sinψ, cosψ = sincos(ψ)
        sinθ, cosθ = sincos(θ)
        sinϕ, cosϕ = sincos(ϕ)
        tanψ = sinψ / cosψ
        tanθ = sinθ / cosθ
        tanϕ = sinϕ / cosϕ
        sinp, cosp = sincos(p)
        sinq, cosq = sincos(q)
        sinr, cosr = sincos(r)
        tanp = sinp / cosp
        tanq = sinq / cosq
        tanr = sinr / cosr

        # Using a StaticArray here doesn't work with FastChain and the rest.
        [
            sinψ,
            cosψ,
            tanψ,
            sinθ,
            cosθ,
            tanθ,
            sinϕ,
            cosϕ,
            tanϕ,
            sinp,
            cosp,
            tanp,
            sinq,
            cosq,
            tanq,
            sinr,
            cosr,
            tanr,
            x^2 + y^2 + z^2,
            ẋ^2 + ẏ^2 + ż^2,
            x,
            y,
            z,
            ψ % twopi,
            θ % twopi,
            ϕ % twopi,
            ẋ,
            ẏ,
            ż,
            p % twopi,
            q % twopi,
            r % twopi,
        ]
    end

    dynamics, cost, sample_x0, observation
end

end


T = 25.0

dynamics, cost, sample_x0, obs = QuadrotorEnv.env(floatT, 9.8f0, 1, 1, 1, 1)
x0 = sample_x0()

num_hidden = 64
policy = FastChain(
    (x, _) -> obs(x),
    FastDense(32, num_hidden, tanh),
    FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, 4),
)

function aug_dynamics(z, policy_params, t)
    x = @view z[2:end]
    u = policy(x, policy_params)
    vcat(cost(x, u), dynamics(x, u))
    # [x' * x + u' * u; u]
end

function aug_dynamics!(dz, z, policy_params, t)
    x = @view z[2:end]
    u = policy(x, policy_params)
    dz[1] = cost(x, u)
    # Note that dynamics!(dz[2:end], x, u) breaks Zygote :(
    dz[2:end] = dynamics(x, u)
end

function loss_pullback(x0, policy_params)
    z0 = vcat(0.0, x0)
    fwd_sol = solve(
        ODEProblem(aug_dynamics!, z0, (0, T), policy_params),
        VCABM(),
        u0 = z0,
        p = policy_params,
        abstol = 1e-3,
        reltol = 1e-3,
    )

    function _adjoint_solve(g_zT, sensealg; kwargs...)
        solve(
            ODEAdjointProblem(fwd_sol, sensealg, (out, x, p, t, i) -> (out[:] = g_zT), [T]),
            VCABM();
            kwargs...,
        )
    end

    # This is the pullback using the augmented system and a discrete
    # gradient input at time T. Alternatively one could use the continuous
    # adjoints on the non-augmented system although that seems to be slower.
    function pullback(g_zT, sensealg::BacksolveAdjoint)
        _adjoint_solve(
            g_zT,
            sensealg,
            dense = false,
            save_everystep = false,
            save_start = false,
            abstol = 1e-3,
            reltol = 1e-3,
        )
        # Not bothering to slice out the gradient from the results of the
        # adjoint solve; just trying to measure performance.
    end
    function pullback(g_zT, sensealg::InterpolatingAdjoint)
        _adjoint_solve(
            g_zT,
            sensealg,
            dense = false,
            save_everystep = false,
            save_start = false,
            abstol = 1e-3,
            reltol = 1e-3,
        )
    end
    function pullback(g_zT, sensealg::QuadratureAdjoint)
        # See https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/quadrature_adjoint.jl#L173.
        # This is 75% of the time and allocs of the pullback. quadgk is
        # actually lightweight relatively speaking.
        _adjoint_solve(
            g_zT,
            sensealg,
            save_everystep = true,
            save_start = true,
            abstol = 1e-3,
            reltol = 1e-3,
        )
        # Skip the whole quadrature bit, only measuring the adjoint solve.
    end

    fwd_sol, pullback
end

policy_params = initial_params(policy)

@info "forward"
@btime loss_pullback(x0, policy_params)

fwd_sol, vjp = loss_pullback(x0, policy_params)
@show fwd_sol.destats.nf
g = vcat(1, zero(x0))

@info "BacksolveAdjoint"
@btime vjp(g, BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)))
@info "InterpolatingAdjoint"
@btime vjp(g, InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
@info "QuadratureAdjoint"
@btime vjp(g, QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)))
nothing
