"""https://github.com/SciML/DifferentialEquations.jl/issues/681

Currently failing"""

import DifferentialEquations: Tsit5, DynamicalODEProblem, solve

v0 = zeros(2)
x0 = ones(2)

aug_dyn_v(v_aug, x_aug, p, t) = zeros(3)
aug_dyn_x(v_aug, x_aug, p, t) = begin
    v = v_aug[2:end]
    x = x_aug[2:end]
    # We accumulate a "cost" metric in x[1], and use x[2:end] for the actual state.
    [sum(x .^ 2); v]
end

# automatic solver selection => loss = 0.0
solvealg = nothing

# Tsit5 => loss > 0.0 (as expected)
# solvealg = Tsit5()

sol = solve(
    DynamicalODEProblem(
        aug_dyn_v,
        aug_dyn_x,
        [0.0; v0],
        [0.0; x0],
        (0.0, 1.0),
    ),
    solvealg,
)

loss = sol.u[end].x[2][1]
@assert loss > 0
