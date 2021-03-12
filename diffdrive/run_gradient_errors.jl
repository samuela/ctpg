include("common.jl")
include("../utils.jl")

import DifferentialEquations: Tsit5, Vern9
import DiffEqFlux: FastChain, FastDense, initial_params, ODEProblem, solve
import Random: seed!
import Zygote
import DiffEqSensitivity:
    InterpolatingAdjoint, BacksolveAdjoint, QuadratureAdjoint, ODEAdjointProblem
import ThreadPools: qmap
import JLSO
import Utils: force

seed!(123)

# Float32 is representative of the numerical accurary available on GPUs and the
# vast majority of neural network frameworks. Float64 is rarely supported.
floatT = Float32
T = 10.0
num_samples = 2500

num_hidden = 32
policy = FastChain(
    FastDense(7, num_hidden, tanh),
    FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, 2),
)

dynamics, cost, sample_x0, obs = DiffDriveEnv.diffdrive_env(floatT, 1.0, 0.5)

function policy_dynamics!(dx, x, policy_params, t)
    u = policy(obs(x), policy_params)
    dx .= dynamics(x, u)
end

function cost_functional(x, policy_params, t)
    cost(x, policy(obs(x), policy_params))
end

# See https://github.com/SciML/DiffEqSensitivity.jl/issues/302 for context.
dcost_tuple = (
    (out, u, p, t) -> begin
        ū, _, _ = Zygote.gradient(cost_functional, u, p, t)
        out .= ū
    end,
    (out, u, p, t) -> begin
        _, p̄, _ = Zygote.gradient(cost_functional, u, p, t)
        out .= p̄
    end,
)

function gold_standard_gradient(x0, policy_params)
    # Actual/gold standard evaluation. Using high-fidelity Vern9 method with
    # small tolerances. We want to use Float64s for maximum accuracy. Also 1e-14
    # is recommended as the minimum allowable tolerance here: https://docs.sciml.ai/stable/basics/faq/#How-to-get-to-zero-error-1.
    x0_f64 = convert(Array{Float64}, x0)
    policy_params_f64 = convert(Array{Float64}, policy_params)
    fwd_sol = solve(
        ODEProblem(policy_dynamics!, x0, (0, T), policy_params),
        Vern9(),
        u0 = x0_f64,
        p = policy_params_f64,
        abstol = 1e-14,
        reltol = 1e-14,
    )
    # Note that specifying dense = false is essential for getting acceptable
    # performance. save_everystep = false is another small win.
    bwd_sol = solve(
        ODEAdjointProblem(
            fwd_sol,
            InterpolatingAdjoint(),
            cost_functional,
            nothing,
            dcost_tuple,
        ),
        Vern9(),
        dense = false,
        save_everystep = false,
        abstol = 1e-14,
        reltol = 1e-14,
    )
    @assert typeof(fwd_sol.u) == Array{Array{Float64,1},1}
    @assert typeof(bwd_sol.u) == Array{Array{Float64,1},1}

    # Note that the backwards solution includes the gradient on x0, as well as
    # policy_params. The full ODESolution things can't be serialized easily
    # since they `policy_dynamics!` and shit...
    (xT = fwd_sol.u[end], g = bwd_sol.u[end])
end

function eval_interp(x0, policy_params, abstol, reltol)
    fwd_sol = solve(
        ODEProblem(policy_dynamics!, x0, (0, T), policy_params),
        Tsit5(),
        u0 = x0,
        p = policy_params,
        abstol = abstol,
        reltol = reltol,
    )
    bwd_sol = solve(
        ODEAdjointProblem(
            fwd_sol,
            InterpolatingAdjoint(),
            cost_functional,
            nothing,
            dcost_tuple,
        ),
        Tsit5(),
        dense = false,
        save_everystep = false,
        abstol = abstol,
        reltol = reltol,
    )
    @assert typeof(fwd_sol.u) == Array{Array{floatT,1},1}
    @assert typeof(bwd_sol.u) == Array{Array{floatT,1},1}

    # Note that g includes the x0 gradient and the gradient on parameters.
    # We do exactly as many f calls as there are function calls in the forward
    # pass, and in the backward pass we don't need to call f, but instead we
    # call ∇f.
    (
        xT = fwd_sol.u[end],
        g = bwd_sol.u[end],
        nf = fwd_sol.destats.nf,
        n∇f = bwd_sol.destats.nf,
    )
end

function eval_backsolve(x0, policy_params, abstol, reltol, checkpointing)
    x_dim = length(x0)

    fwd_sol = solve(
        ODEProblem(policy_dynamics!, x0, (0, T), policy_params),
        Tsit5(),
        u0 = x0,
        p = policy_params,
        abstol = abstol,
        reltol = reltol,
    )
    bwd_sol = solve(
        ODEAdjointProblem(
            fwd_sol,
            BacksolveAdjoint(checkpointing = checkpointing),
            cost_functional,
            nothing,
            dcost_tuple,
        ),
        Tsit5(),
        dense = false,
        save_everystep = false,
        abstol = abstol,
        reltol = reltol,
    )
    @assert typeof(fwd_sol.u) == Array{Array{floatT,1},1}
    @assert typeof(bwd_sol.u) == Array{Array{floatT,1},1}

    # In the backsolve adjoint, the last x_dim dimensions are for the
    # reconstructed x state.
    # When running the backsolve adjoint we have additional f evaluations every
    # step of the backwards pass, since we need -f to reconstruct the x path.
    (
        xT = fwd_sol.u[end],
        g = bwd_sol.u[end][1:end-x_dim],
        nf = fwd_sol.destats.nf + bwd_sol.destats.nf,
        n∇f = bwd_sol.destats.nf,
    )
end

function euler_with_cost(x0, policy_params, dt, num_steps)
    x = x0
    cost_accum = 0.0
    for _ = 1:num_steps
        u = policy(obs(x), policy_params)
        cost_accum += dt * cost(x, u)
        x += dt * dynamics(x, u)
    end
    cost_accum
end

function eval_euler_bptt(x0, policy_params, dt)
    # Julia seems to do auto-rounding with floor when doing 1:num_steps. That's
    # fine for our purposes.
    num_steps = T / dt
    g_x0, g_θ = Zygote.gradient(
        (x0, θ) -> euler_with_cost(x0, θ, dt, num_steps),
        x0,
        policy_params,
    )
    # We require gradients on both x0 and θ, vcat'd together. For some reason
    # the DifferentialEquations.jl convention is to negate the x0 gradients.
    (g = vcat(-g_x0, g_θ), nf = num_steps, n∇f = num_steps)
end

# See https://github.com/SciML/DiffEqSensitivity.jl/issues/304 for why this
# doesn't work.
# function eval_quadrature(x0, policy_params, abstol, reltol)
#     fwd_sol = solve(
#         ODEProblem(policy_dynamics!, x0, (0, T), policy_params),
#         Tsit5(),
#         u0 = x0,
#         p = policy_params,
#         abstol = abstol,
#         reltol = reltol,
#     )
#     bwd_sol = solve(
#         ODEAdjointProblem(fwd_sol, QuadratureAdjoint(), cost_functional),
#         Tsit5(),
#         # dense = false,
#         # save_everystep = false,
#         abstol = abstol,
#         reltol = reltol,
#     )
#     @assert typeof(fwd_sol.u) == Array{Array{floatT,1},1}
#     @assert typeof(bwd_sol.u) == Array{Array{floatT,1},1}
#
#     # The way to do this is defined here: https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/quadrature_adjoint.jl#L173
#
#     # TODO: figure out how to get the number of function evals from quadgk...
#
#     # (fwd = fwd_sol, bwd = estbwd_sol, g = estbwd_sol.u[end][1:end-x_dim])
# end

# init_conditions = [(sample_x0(), initial_params(policy)) for _ = 1:num_samples]
x0_samples = [sample_x0() for _ = 1:num_samples]
θ_samples = [convert(Array{floatT}, 0.1 * initial_params(policy)) for _ = 1:num_samples]

# Must be before plot_results! since that depends on this...
@info "gold standard"
# @benchmark gold_standard_gradient(x0_samples[1], θ_samples[1])
@time gold_standard_results = qmap(
    force,
    [
        () -> gold_standard_gradient(x0, θ)
        for (x0, θ) in zip(x0_samples, θ_samples)
    ],
)

@info "euler bptt"
# 1e-5 is too small and exhausts the memory available.
@time euler_bptt_results = qmap(
    force,
    [
        () -> eval_euler_bptt(x0, θ, dt)
        for
        (x0, θ, dt) in
        zip(x0_samples, θ_samples, 10 .^ range(-4, 0, length = num_samples))
    ],
)

# Absolute error tolerances should generally be smaller than relative
# tolerances. Do the smallest ones first in order to hit memory issues sooner
# rather than later.
# reltols = 0.1.^(0:9)
reltols = 10 .^ range(-9, 0, length = num_samples)
abstols = 1e-3 * reltols

@info "interp"
# @benchmark eval_interp(x0_samples[1], θ_samples[1], 1e-12, 1e-9)
@time interp_results = qmap(
    force,
    [
        () -> eval_interp(x0, θ, atol, rtol)
        for
        (x0, θ, rtol, atol) in zip(x0_samples, θ_samples, reltols, abstols)
    ],
)

@info "backsolve"
@time backsolve_results = qmap(
    force,
    [
        () -> eval_backsolve(x0, θ, atol, rtol, false)
        for
        (x0, θ, rtol, atol) in zip(x0_samples, θ_samples, reltols, abstols)
    ],
)

@info "backsolve with checkpoints"
@time backsolve_checkpointing_results = qmap(
    force,
    [
        () -> eval_backsolve(x0, θ, atol, rtol, true)
        for
        (x0, θ, rtol, atol) in zip(x0_samples, θ_samples, reltols, abstols)
    ],
)

# See https://github.com/SciML/DiffEqSensitivity.jl/issues/304.
# quadrature_results = [
#     [eval_quadrature(x0, θ, atol, rtol) for (x0, θ) in init_conditions] for (atol, rtol) in zip(abstols, reltols)
# ]

JLSO.save(
    "diffdrive_gradient_error_results.jlso",
    :gold_standard_results => gold_standard_results,
    :euler_bptt_results => euler_bptt_results,
    :interp_results => interp_results,
    :backsolve_results => backsolve_results,
    :backsolve_checkpointing_results => backsolve_checkpointing_results,
)
