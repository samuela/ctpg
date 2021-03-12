"""An differential drive robot environment.

See http://planning.cs.uiuc.edu/node659.html.
"""

module DiffDriveEnv

function env(floatT, wheelbase, wheel_radius)
    twopi = convert(floatT, 2π)

    function dynamics(state, u)
        x, y, θ, ω_l, ω_r = state
        sinθ, cosθ = sincos(θ)
        # clippy(z, dz) =
        #     if z < -1
        #         max(dz, 0)
        #     elseif z > 1
        #         min(dz, 0)
        #     else
        #         dz
        #     end
        [
            wheel_radius / 2 * (ω_l + ω_r) * cosθ,
            wheel_radius / 2 * (ω_l + ω_r) * sinθ,
            wheel_radius / wheelbase * (ω_r - ω_l),
            # clippy(ω_l, u[1]),
            # clippy(ω_r, u[2]),
            u[1],u[2]
        ]
    end

    function cost(state, u)
        x, y, θ, ω_l, ω_r = state
        x^2 + y^2 + 0.1 * (ω_l^2 + ω_r^2 + u[1]^2 + u[2]^2)
    end

    function sample_x0()
        [
            rand(floatT) * 10 - 5,
            rand(floatT) * 10 - 5,
            rand(floatT) * twopi,
            # randn(floatT),
            # randn(floatT),
            0,
            0,
        ]::Array{floatT}
    end

    function observation(state)
        x, y, θ, ω_l, ω_r = state
        sinθ, cosθ = sincos(θ)
        [x, y, θ % twopi, ω_l, ω_r, sinθ, cosθ]
        # [x, y, θ % twopi, ω_l, ω_r, sinθ, cosθ, x^2, y^2, ω_l^2, ω_r^2]
    end

    dynamics, cost, sample_x0, observation
end

end
