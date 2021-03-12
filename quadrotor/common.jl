"""Quadrotor

See https://www.kth.se/polopoly_fs/1.588039.1550155544!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf,
especially equations 2.23-2.25.
"""

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
            q * sinϕ / cosθ + r * cosϕ / cosθ,
            q * cosϕ - r * sinϕ,
            p + q * sinϕ * tanθ + r * cosϕ * tanθ,
            g_1_7 * u[1],
            g_1_8 * u[1],
            gravity + g_1_9 * u[1],
            (Iy - Iz) / Ix * q * r + u[2] / Ix,
            (Iz - Ix) / Iy * p * r + u[3] / Iy,
            (Ix - Iy) / Iz * p * q + u[4] / Iz,
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
