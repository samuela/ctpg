import DifferentialEquations: Tsit5
import Flux: ADAM
import Flux.Data: DataLoader
import DiffEqFlux:
    FastChain, FastDense, initial_params, sciml_train, ODEProblem, solve
import Random: seed!, randn
import Plots: plot
import Statistics: mean
import Zygote
using Optim: LBFGS, BFGS, Fminbox
import DiffEqSensitivity: InterpolatingAdjoint
import LinearAlgebra: I
import LineSearches

include("pendulum.jl")

seed!(123)

T = 5.0
batch_size = 1
num_hidden = 64
policy = FastChain(
    FastDense(4, num_hidden, tanh),
    FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, 1),
    # (x, _) -> 2 * x,
)
# policy = FastDense(4, 1) # linear policy

# The model weights are destructured into a vector of parameters
init_policy_params = initial_params(policy)

dynamics, cost, sample_x0 = Pendulum.pendulum_env(1, 1, 9.8, 0)

function preprocess(x)
    θ, θ_dot = x
    sinθ, cosθ = sincos(θ)
    [θ, θ_dot, sinθ, cosθ]
end

function aug_dynamics!(dz, z, policy_params, t)
    x = z[2:end]
    u = policy(preprocess(x), policy_params)[1]
    dz[1] = cost(x, u)
    # Note that dynamics!(dz[2:end], x, u) breaks Zygote :(
    dz[2:end] = dynamics(x, u)
end

# @benchmark aug_dynamics!(rand(3), rand(3), init_policy_params, 0.0)

### Example rollout.
# x0 = [π - 0.1f0, 0f0]::Array{Float32}
# z0 = [0f0, x0...]::Array{Float32}
# rollout = solve(
#     ODEProblem(aug_dynamics!, z0, (0, T)),
#     Tsit5(),
#     u0 = z0,
#     p = init_policy_params,
# )
# tspan = 0:0.05:T
# plot(tspan, hcat(rollout.(tspan)...)', label = ["cost" "θ" "θ dot"])

function loss(policy_params, data...)
    # TODO: use the ensemble thing
    mean([
        begin
            z0 = [0f0, x0...]
            rollout = solve(
                ODEProblem(aug_dynamics!, z0, (0, T), policy_params),
                Tsit5(),
                u0 = z0,
                p = policy_params,
                sensealg = InterpolatingAdjoint(),
            )
            Array(rollout)[1, end]
        end for x0 in data
    ])
end

callback = function (policy_params, loss_val)
    println("Loss $loss_val")
    z0_bottom = [0f0, 0f0, 0f0]::Array{Float32}
    rollout = solve(
        ODEProblem(aug_dynamics!, z0_bottom, (0, T), policy_params),
        Tsit5(),
        u0 = z0_bottom,
        p = policy_params,
    )
    tspan = 0:0.05:T
    rollout_arr = hcat(rollout.(tspan)...)
    display(plot(
        tspan,
        [cos.(rollout_arr[2, :]), rollout_arr[3, :]],
        label = ["cos(θ)" "θ dot"],
        title = "Swing up cost: $(rollout_arr[1, end])",
    ))

    false
end

data = DataLoader([sample_x0() for _ = 1:1_000_000], batchsize = batch_size)
# res1 = sciml_train(loss, init_policy_params, ADAM(), data, cb = callback)
res1 = sciml_train(
    loss,
    init_policy_params,
    # BFGS(
    #     alphaguess = LineSearches.InitialStatic(alpha = 0.1),
    #     linesearch = LineSearches.Static(),
    # ),
    LBFGS(
        alphaguess = LineSearches.InitialStatic(alpha = 0.01),
        linesearch = LineSearches.Static(),
    ),
    data,
    cb = callback,
    iterations = length(data),
    allow_f_increases = true,
    x_abstol = NaN,
    x_reltol = NaN,
    f_abstol = NaN,
    f_reltol = NaN,
    g_abstol = NaN,
    g_reltol = NaN,
)

# @profile res1 =
#     sciml_train(loss, init_policy_params, ADAM(), [first(data)], cb = callback)

# 1.180s median
# @benchmark Zygote.gradient(loss, init_policy_params, first(data)...)
# @benchmark Zygote.gradient(loss, init_policy_params, first(data)...)

# 1.192s median
# @benchmark sciml_train(loss, init_policy_params, ADAM(), data, maxiters = 1)

# begin
#     import LineSearches
#     opt = LBFGS(
#         alphaguess = LineSearches.InitialStatic(alpha = 0.1),
#         linesearch = LineSearches.Static(),
#     )
#     @timev sciml_train(loss, init_policy_params, opt, data, maxiters = 1)
#     @benchmark sciml_train(loss, init_policy_params, opt, data, maxiters = 1)
# end
