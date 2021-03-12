import JLSO
import Plots
import ProgressMeter: @showprogress
import Random
import LinearAlgebra: norm

Random.seed!(123)
ENV["GKSwstype"] = "nul"

results = JLSO.load("quadrotor_train_results.jlso")
# ppg_params = results[:interp_results].policy_params_per_iter
euler_params = results[:euler_results].policy_params_per_iter
# @assert size(ppg_params) == size(euler_params)
num_iter = size(ppg_params, 1)

Plots.savefig(
    Plots.plot(
        1:num_iter,
        [results[:euler_results].loss_per_iter, results[:euler_results].loss_per_iter],
        label = ["Euler BPTT" "PPG (ours)"],
        xlabel = "Iteration",
        ylabel = "Loss",
    ),
    "quadrotor_loss_per_iter.pdf",
)

begin
    p = Plots.plot(xlabel = "Number of function evaluations", ylabel = "Loss")
    Plots.plot!(
        cumsum(results[:euler_results].nf_per_iter + results[:euler_results].n∇f_per_iter),
        results[:euler_results].loss_per_iter,
        label = "Euler BPTT",
    )
    # Plots.plot!(
    #     cumsum(
    #         results[:interp_results].nf_per_iter + results[:interp_results].n∇f_per_iter,
    #     ),
    #     results[:interp_results].loss_per_iter,
    #     label = "PPG (ours)",
    # )
    Plots.savefig(p, "quadrotor_loss_per_nf.pdf")
end

Plots.savefig(
    Plots.plot(
        map(mean, eachrow(abs.(euler_results.policy_params_per_iter))),
        label = "mean abs of each weight",
    ),
    "quadrotor_weights_per_iter.pdf",
)

Plots.savefig(
    Plots.plot(map(norm, eachrow(euler_results.g_per_iter)), label = "norm of gradients"),
    "quadrotor_grads_per_iter.pdf",
)

Plots.savefig(
    Plots.plot(
        map(mean, eachrow(abs.(euler_results.g_per_iter))),
        label = "mean abs of gradients",
        yaxis = :log10,
    ),
    "quadrotor_grads_per_iter.pdf",
)

# euler_dt_per_iter =
#     T ./ (
#         cumsum(
#             (interp_results.nf_per_iter + interp_results.n∇f_per_iter) / batch_size / 2,
#         ) ./ (1:num_iter)
#     )
# Plots.savefig(Plots.plot(euler_dt_per_iter, label="Average Euler dt over time"), "deleteme.pdf")

# NOTE: Make sure this is the same as in train.jl!
dynamics, cost, sample_x0, obs = quadrotorEnv.quadrotor_env(floatT, 1.0f0, 0.5f0)
learned_policy_goodies = ppg_goodies(dynamics, cost, policy, T)
x0_test_batch = [sample_x0() for _ = 1:10]
# num_hidden = 32
# policy = FastChain(
#     (x, _) -> obs(x),
#     FastDense(7, num_hidden, tanh),
#     FastDense(num_hidden, num_hidden, tanh),
#     FastDense(num_hidden, 2),
# )

function rollout(x0, θ)
    sol = solve(
        ODEProblem((x, p, t) -> dynamics(x, policy(x, θ)), x0, (0, T)),
        Tsit5(),
        saveat = 0.1,
    )
    xs = [s[1] for s in sol.u]
    ys = [s[2] for s in sol.u]
    (xs, ys)
end

# @time ppg_trajs = qmap(collect(eachrow(ppg_params))) do θ
#     [rollout(x0, θ) for x0 in x0_test_batch]
# end
@time euler_trajs = qmap(collect(eachrow(euler_params))) do θ
    [rollout(x0, θ) for x0 in x0_test_batch]
end

# stills = [5000, 7500, 10000]

# anim = Plots.Animation()
# @showprogress for iter = 1:size(ppg_params, 1)
#     p = Plots.plot(
#         title = "Iteration $iter",
#         legend = true,
#         aspect_ratio = :equal,
#         xlims = (-7.5, 7.5),
#         ylims = (-7.5, 7.5),
#     )
#     for (xs, ys) in euler_trajs[iter]
#         Plots.plot!(xs, ys, color = :blue, linestyle = :dash, label = nothing)
#     end
#     for (xs, ys) in ppg_trajs[iter]
#         Plots.plot!(xs, ys, color = :red, label = nothing)
#     end
#     Plots.scatter!(
#         [x0[1] for x0 in x0_test_batch],
#         [x0[2] for x0 in x0_test_batch],
#         color = :grey,
#         markersize = 5,
#         label = "Initial conditions",
#     )
#     Plots.plot!([], color = :blue, linestyle = :dash, label = "Euler BPTT")
#     Plots.plot!([], color = :red, label = "PPG (ours)")
#     Plots.frame(anim)
#     if iter in stills
#         Plots.savefig(p, "quadrotor_still_$iter.pdf")
#     end
# end
# Plots.mp4(anim, "quadrotor.mp4")
