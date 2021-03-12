"""Test that we can handle multiple contacts."""

include("../ppg_toi.jl")

import DifferentialEquations
import UnicodePlots: lineplot

n_objects = 2
v_dynamics(v, x, u) = zeros(n_objects)
x_dynamics(v, x, u) = v
cost(v, x, u) = 0
policy(v, x, t, p) = 0
toi_affect(old_v, old_x, dt) = begin
    # We want to make sure that we only register for negative velocities.
    tois = [-x / min(v, -eps()) for (x, v) in zip(old_x, old_v)]
    impact_velocity = 0 # [0, 0] for mass_spring

    new_v = [if toi < dt impact_velocity else v end for (toi, v) in zip(tois, old_v)]
    new_x = old_x + min.(tois, dt) .* old_v + max.(dt .- tois, 0) .* new_v
    (new_v, new_x)
end

T = 1.0
loss_pullback = ppg_toi_goodies(
    v_dynamics,
    x_dynamics,
    cost,
    policy,
    TOIStuff([(v, x) -> x[i] for i in 1:n_objects], toi_affect, 1e-3),
    T
)

v0 = [-1, -1]
x0 = [0.25, 0.5]
sol, _ = loss_pullback(v0, x0, zeros(), nothing, Dict())

@assert length(sol.solutions) == 3

ts = 0:0.01:T
# Don't forget the first is for the cost!
lineplot(ts, [z[2] for z in sol.(ts)]) |> show
lineplot(ts, [z[3] for z in sol.(ts)]) |> show
