module Pendulum

import DifferentialEquations: Tsit5, ODEProblem, solve
import Random: randn
import Plots: plot

function pendulum_env(mass, length, gravity, friction)
    function dynamics(x, u)
        θ, θ_dot = x
        θ_dotdot =
            -gravity / length * sin(θ) - friction * θ_dot +
            u / (mass * length^2)

        # if θ > 8
        #     θ_dotdot = min(θ_dotdot, 0)
        # elseif θ < -8
        #     θ_dotdot = max(θ_dotdot, 0)
        # end
        [θ_dot, θ_dotdot]
    end

    function cost(x, u)
        θ, θ_dot = x
        # Unlike the python version mod in julia can return negative values.
        θ = abs(θ % (2 * pi))
        (θ - π)^2 + 0.1 * (θ_dot^2) + 0.001 * (u^2)
    end

    function sample_x0()
        θ = rand(Float32) * 2 * π
        θ_dot = rand(Float32) * 2 - 1
        [θ, θ_dot]
    end

    dynamics, cost, sample_x0
end

function example_rollout()
    dynamics, cost, sample_x0 = pendulum_env(1, 1, 9.8, 0)

    ### Example rollout.
    T = 5.0
    x0 = [π - 0.1f0, 0f0]::Array{Float32}
    rollout = solve(
        ODEProblem((x, _, _) -> dynamics(x, 0.0), x0, (0, T)),
        Tsit5(),
        u0 = x0,
        saveat = 0:0.05:T,
    )
    plot(rollout.t, Array(rollout)', label = ["θ" "θ dot"])
end

end
