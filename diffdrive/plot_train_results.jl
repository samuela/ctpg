import JLSO
import Plots
import ProgressMeter: @showprogress
import Random
import LinearAlgebra: norm
import Statistics: mean
import DiffEqFlux: FastChain, FastDense
import DifferentialEquations: Tsit5

include("common.jl")
include("../ppg.jl")

Random.seed!(123)
ENV["GKSwstype"] = "nul"

results = JLSO.load("diffdrive_train_results.jlso")
ppg_params = results[:interp_results].policy_params_per_iter
euler_params = results[:euler_results].policy_params_per_iter
@assert size(ppg_params) == size(euler_params)
num_iter = size(ppg_params, 1)

Plots.savefig(
    Plots.plot(
        1:num_iter,
        [results[:euler_results].loss_per_iter, results[:interp_results].loss_per_iter],
        # Not a typo: `label` needs to be 1x2 for some reason...
        label = ["Euler BPTT" "CTPG (ours)"],
        xlabel = "Iteration",
        ylabel = "Loss",
    ),
    "diffdrive_loss_per_iter.pdf",
)

begin
    p = Plots.plot(xlabel = "Number of function evaluations", ylabel = "Loss")
    Plots.plot!(
        cumsum(results[:euler_results].nf_per_iter + results[:euler_results].n∇ₓf_per_iter + results[:euler_results].n∇ᵤf_per_iter),
        results[:euler_results].loss_per_iter,
        label = "Euler BPTT",
    )
    Plots.plot!(
        cumsum(
            results[:interp_results].nf_per_iter + results[:interp_results].n∇ₓf_per_iter + results[:interp_results].n∇ᵤf_per_iter,
        ),
        results[:interp_results].loss_per_iter,
        label = "CTPG (ours)",
    )
    Plots.savefig(p, "diffdrive_loss_per_nf.pdf")
end

Plots.savefig(
    Plots.plot(
        map(mean, eachrow(abs.(results[:interp_results].policy_params_per_iter))),
        label = "mean abs of each weight",
    ),
    "diffdrive_weights_per_iter.pdf",
)

Plots.savefig(
    Plots.plot(map(norm, eachrow(results[:interp_results].g_per_iter)), label = "norm of gradients"),
    "diffdrive_grads_per_iter.pdf",
)

Plots.savefig(
    Plots.plot(
        map(mean, eachrow(abs.(results[:interp_results].g_per_iter))),
        label = "mean abs of gradients",
        yaxis = :log10,
    ),
    "diffdrive_grads_per_iter.pdf",
)

# euler_dt_per_iter =
#     T ./ (
#         cumsum(
#             (interp_results.nf_per_iter + interp_results.n∇f_per_iter) / batch_size / 2,
#         ) ./ (1:num_iter)
#     )
# Plots.savefig(Plots.plot(euler_dt_per_iter, label="Average Euler dt over time"), "deleteme.pdf")

x0_test_batch = [sample_x0() for _ = 1:10]

# NOTE: Make sure this is exactly the same as in train.jl when the results were
# gathered!
T = 5.0
floatT = Float32
dynamics, cost, sample_x0, obs = DiffDriveEnv.env(floatT, 1.0f0, 0.5f0)
num_hidden = 32
policy = FastChain(
    (x, _) -> obs(x),
    FastDense(7, num_hidden, tanh),
    FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, 2),
)
learned_policy_goodies = ppg_goodies(dynamics, cost, (x, _, pp) -> policy(x, pp), T)

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

@time ppg_trajs = qmap(collect(eachrow(ppg_params))) do θ
    [rollout(x0, θ) for x0 in x0_test_batch]
end
@time euler_trajs = qmap(collect(eachrow(euler_params))) do θ
    [rollout(x0, θ) for x0 in x0_test_batch]
end

stills = [5000, 7500, 10000]

anim = Plots.Animation()
@showprogress for iter = 1:size(ppg_params, 1)
    p = Plots.plot(
        title = "Iteration $iter",
        legend = true,
        aspect_ratio = :equal,
        xlims = (-7.5, 7.5),
        ylims = (-7.5, 7.5),
    )
    for (xs, ys) in euler_trajs[iter]
        Plots.plot!(xs, ys, color = :blue, linestyle = :dash, label = nothing)
    end
    for (xs, ys) in ppg_trajs[iter]
        Plots.plot!(xs, ys, color = :red, label = nothing)
    end
    Plots.scatter!(
        [x0[1] for x0 in x0_test_batch],
        [x0[2] for x0 in x0_test_batch],
        color = :grey,
        markersize = 5,
        label = "Initial conditions",
    )
    Plots.plot!([], color = :blue, linestyle = :dash, label = "Euler BPTT")
    Plots.plot!([], color = :red, label = "CTPG (ours)")
    Plots.frame(anim)
    if iter in stills
        Plots.savefig(p, "diffdrive_still_$iter.pdf")
    end
end
Plots.mp4(anim, "diffdrive.mp4")
