"""Train a not-failing policy on a linear environment."""

include("common.jl")
include("../utils.jl")
include("../ppg.jl")

import DifferentialEquations: Tsit5
import Flux
import Flux: Momentum
import DiffEqFlux:
    FastChain, FastDense, initial_params, sciml_train, ODEProblem, solve
import Random: seed!
import DiffEqSensitivity: InterpolatingAdjoint
import LinearAlgebra: I
import ControlSystems
import ProgressMeter

seed!(123)

floatT = Float32
x_dim = 2
T = 25.0
num_hidden = 32
policy = FastChain(
    FastDense(x_dim, num_hidden, tanh),
    # FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, x_dim),
)
# policy = FastDense(x_dim, x_dim) # linear policy

A = Matrix{floatT}(0 * I, x_dim, x_dim)
B = Matrix{floatT}(I, x_dim, x_dim)
Q = Matrix{floatT}(I, x_dim, x_dim)
R = Matrix{floatT}(I, x_dim, x_dim)
dynamics, cost, sample_x0 = LinearEnv.linear_env(floatT, x_dim, A, B, Q, R)

K = ControlSystems.lqr(A, B, Q, R)

ppg = ppg_goodies(dynamics, cost, (x, _, params) -> policy(x, params), T)
lqr_ppg = ppg_goodies(dynamics, cost, (x, _, K) -> -K * x, T)

num_iters = 1000
batch_size = 1
learned_loss_per_iter = fill(NaN, num_iters)
lqr_loss_per_iter = fill(NaN, num_iters)
policy_params = initial_params(policy) * 0.1
opt = Momentum(0.0001)
progress = ProgressMeter.Progress(num_iters)
for iter = 1:num_iters
    # x0_batch = [sample_x0() for _ = 1:batch_size]
    x0_batch = [ones(x_dim)]

    loss, g, info =
        ppg.ez_loss_and_grad_many(x0_batch, policy_params, Tsit5(), InterpolatingAdjoint())
    lqr_loss, _, _ =
        lqr_ppg.ez_loss_and_grad_many(x0_batch, K, Tsit5(), InterpolatingAdjoint())

    Flux.Optimise.update!(opt, policy_params, g)
    learned_loss_per_iter[iter] = loss
    lqr_loss_per_iter[iter] = lqr_loss
    ProgressMeter.next!(
        progress;
        showvalues = [
            (:iter, iter),
            (:excess_loss, loss - lqr_loss),
            (:nf, info.nf / batch_size),
            (:n∇ₓf, info.n∇ₓf / batch_size),
            (:n∇ᵤf, info.n∇ᵤf / batch_size),
        ],
    )
end

# import Plots: plot
# plot(learned_loss_per_iter - lqr_loss_per_iter)
