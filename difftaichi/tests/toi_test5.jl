"""Same as test 4 except that we test multiple, simultaneous contacts."""

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
    tois = [-old_x[i] / min(old_v[i], -eps()) for i in 1:n_objects]
    impact_velocities = -old_v

    # Can't use zip. See https://github.com/FluxML/Zygote.jl/issues/221.
    new_v = [if tois[i] < dt impact_velocities[i] else old_v[i] end for i in 1:n_objects]
    new_x = old_x + min.(tois, dt) .* old_v + max.(dt .- tois, 0) .* new_v
    (new_v, new_x)
end

T = 1.23
loss_pullback = ppg_toi_goodies(
    v_dynamics,
    x_dynamics,
    cost,
    policy,
    TOIStuff([(v, x) -> x[i] for i in 1:n_objects], toi_affect, 1e-6),
    T
)

v0 = [-1, -1]
x0 = [0.5, 0.5]
sol, pullback = loss_pullback(v0, x0, zeros(), nothing, Dict())
@assert length(sol.solutions) == 2

pb = pullback((zeros(n_objects + 1), [0, 1, 0]), InterpolatingAdjoint())
@assert isapprox(pb.g_z0[1], [0, -T, 0], atol = 1e-3)
@assert isapprox(pb.g_z0[2], [0, -1, 0])
pb = pullback((zeros(n_objects + 1), [0, 0, 1]), InterpolatingAdjoint())
@assert isapprox(pb.g_z0[1], [0, 0, -T], atol = 1e-3)
@assert isapprox(pb.g_z0[2], [0, 0, -1])
