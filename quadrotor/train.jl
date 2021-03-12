"""Train a differential drive policy and create an animation of the training
process displaying its adaptation on a set of paths over time. Note that when
running on a headless machine, the environment variable `GKS_WSTYPE=140`
generally needs to be set. See https://discourse.julialang.org/t/unable-to-display-plot-using-the-repl-gks-errors/12826/16.
"""

include("common.jl")
include("../ppg.jl")

import DifferentialEquations: Tsit5
import Flux
import Flux: ADAM, Momentum, Optimiser
import Flux.Data: DataLoader
import Flux.Optimise: ExpDecay
import DiffEqFlux: FastChain, FastDense, initial_params
import Random: seed!
import Plots
import Statistics: mean
import DiffEqSensitivity: InterpolatingAdjoint, BacksolveAdjoint, QuadratureAdjoint
import JLSO
import Optim: LBFGS
import LineSearches
import ProgressMeter

# BLAS.set_num_threads(1)

# This seeds the `init_policy_params` below, and then gets overridden later.
seed!(123)

floatT = Float32
T = 5.0
num_iters = 10000
batch_size = 32

dynamics, cost, sample_x0, obs = QuadrotorEnv.env(floatT, 9.8f0, 1, 1, 1, 1)

num_hidden = 64
policy = FastChain(
    (x, _) -> obs(x),
    FastDense(32, num_hidden, tanh),
    FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, 4),
)

# linear policy
# policy = FastChain((x, _) -> obs(x), FastDense(7, 2))

init_policy_params = initial_params(policy) * 0.1
learned_policy_goodies = ppg_goodies(dynamics, cost, policy, T)

function run(loss_and_grad)
    # Seed here so that both interp and euler get the same batches.
    seed!(123)

    loss_per_iter = fill(NaN, num_iters)
    policy_params_per_iter = fill(NaN, num_iters, length(init_policy_params))
    g_per_iter = fill(NaN, num_iters, length(init_policy_params))
    nf_per_iter = fill(NaN, num_iters)
    n∇f_per_iter = fill(NaN, num_iters)

    policy_params = deepcopy(init_policy_params)
    batches = [[sample_x0() for _ = 1:batch_size] for _ = 1:num_iters]
    # opt = ADAM()
    opt = Momentum(0.001)
    # opt = Optimiser(ExpDecay(0.001, 0.5, 1000, 1e-5), Momentum(0.001))
    # opt =LBFGS(
    #     alphaguess = LineSearches.InitialStatic(alpha = 0.001),
    #     linesearch = LineSearches.Static(),
    # )
    progress = ProgressMeter.Progress(num_iters)
    for iter = 1:num_iters
        loss, g, info = loss_and_grad(batches[iter], policy_params)
        loss_per_iter[iter] = loss
        policy_params_per_iter[iter, :] = policy_params
        g_per_iter[iter, :] = g
        nf_per_iter[iter] = info.nf
        n∇f_per_iter[iter] = info.n∇f

        clamp!(g, -10, 10)
        Flux.Optimise.update!(opt, policy_params, g)
        ProgressMeter.next!(
            progress;
            showvalues = [
                (:iter, iter),
                (:loss, loss),
                (:nf, info.nf / batch_size),
                (:n∇f, info.n∇f / batch_size),
            ],
        )
    end

    (
        loss_per_iter = loss_per_iter,
        policy_params_per_iter = policy_params_per_iter,
        g_per_iter = g_per_iter,
        nf_per_iter = nf_per_iter,
        n∇f_per_iter = n∇f_per_iter,
    )
end

# @info "Quad"
# interp_results = run(
#     (x0_batch, θ) -> learned_policy_goodies.ez_loss_and_grad_many(
#         x0_batch,
#         θ,
#         QuadratureAdjoint(),
#     ),
# )

# @info "Interp"
# interp_results = run(
#     (x0_batch, θ) -> learned_policy_goodies.ez_loss_and_grad_many(
#         x0_batch,
#         θ,
#         InterpolatingAdjoint(),
#     ),
# )

# @info "Neural ODE"
# neural_ode_results = run(
#     (x0_batch, θ) -> learned_policy_goodies.ez_loss_and_grad_many(
#         x0_batch,
#         θ,
#         BacksolveAdjoint(checkpointing = false),
#     ),
# )

# Divide by two for the forward and backward passes.
# mean_euler_steps =
#     mean((interp_results.nf_per_iter + interp_results.n∇f_per_iter) / batch_size / 2)
# euler_dt = T / mean_euler_steps
euler_dt = 0.05

@info "Euler"
euler_results = run(
    (x0_batch, θ) ->
        learned_policy_goodies.ez_euler_loss_and_grad_many(x0_batch, θ, euler_dt),
)

@info "Dumping results"
JLSO.save(
    "quadrotor_train_results.jlso",
    # :neural_ode_results => neural_ode_results,
    # :interp_results => interp_results,
    :euler_results => euler_results,
)
