import JLSO
import LinearAlgebra: norm
import Statistics: median, mean, std
import Plots

data = JLSO.load("9e0db54_lqr_gradient_error_results.jlso")
backsolve_results = data[:backsolve_results]
backsolve_checkpointing_results = data[:backsolve_checkpointing_results]
interp_results = data[:interp_results]
euler_bptt_results = data[:euler_bptt_results]
gold_standard_results = data[:gold_standard_results]

# 9e0db54 fixes
num_samples = 256
euler_bptt_results_fixed = [
    (dt, merge(res, (g = res.g .* vcat(-ones(2), ones(6)),)))
    for (dt, res) in euler_bptt_results
]

function plot_line!(results_flat, label)
    # An array where rows correspond to each (x0, θ) pair, and columns
    # correspond to each tolerance level. Use the map to drop the descriptive
    # tolerance key value.
    results = reshape(map((t) -> t[end], results_flat), :, num_samples)
    results = collect(eachrow(results))

    nf_calls = [[sol.nf + sol.n∇f for sol in res] for res in results]
    g_errors = [
        [
            norm(gold.g - est.g)
            for (gold, est) in zip(gold_standard_results, res)
        ] for res in results
    ]

    function safe_error_bars(vs)
        vs_median = map(median, vs)
        vs_mean = map(mean, vs)
        vs_std = map(std, vs)
        collect(zip(
            [
                min(ṽ - 1e-12, σ + ṽ - μ)
                for (ṽ, μ, σ) in zip(vs_median, vs_mean, vs_std)
            ],
            vs_std + vs_mean - vs_median,
        ))
    end

    Plots.plot!(
        map(median, nf_calls),
        map(median, g_errors),
        xlabel = "Function evaluations",
        ylabel = "L2 error in the policy gradient",
        xaxis = :log10,
        yaxis = :log10,
        xerror = safe_error_bars(nf_calls),
        yerror = safe_error_bars(g_errors),
        label = label,
    )
end

function plot_scatter!(results_flat, label)
    # An array where rows correspond to each (x0, θ) pair, and columns
    # correspond to each tolerance level. Use the map to drop the descriptive
    # tolerance key value.
    results = reshape(map((t) -> t[end], results_flat), :, num_samples)
    results = collect(eachrow(results))

    nf_calls = [[sol.nf + sol.n∇f for sol in res] for res in results]
    g_errors = [
        [
            norm(gold.g - est.g)
            for (gold, est) in zip(gold_standard_results, res)
        ] for res in results
    ]

    flatten(arrs) = [x for xs in arrs for x in xs]

    Plots.scatter!(
        flatten(nf_calls),
        flatten(g_errors),
        xlabel = "Function evaluations",
        ylabel = "L2 error in the policy gradient",
        xguidefontsize=16,
        yguidefontsize=16,
        xaxis = :log10,
        yaxis = :log10,
        alpha = 0.25,
        markerstrokewidth = 0,
        label = label,
    )
end

function plot_ribbon!(results_flat, label)
    # An array where rows correspond to each (x0, θ) pair, and columns
    # correspond to each tolerance level. Use the map to drop the descriptive
    # tolerance key value.
    results = reshape(map((t) -> t[end], results_flat), :, num_samples)
    results = collect(eachrow(results))

    nf_calls = [[sol.nf + sol.n∇f for sol in res] for res in results]
    g_errors = [
        [
            norm(gold.g - est.g)
            for (gold, est) in zip(gold_standard_results, res)
        ] for res in results
    ]

    Plots.plot!(
        map(median, nf_calls),
        map(median, g_errors),
        xlabel = "Function evaluations",
        ylabel = "L2 error in the policy gradient",
        ribbon = (
            map(median, g_errors) - map(minimum, g_errors),
            map(maximum, g_errors) - map(median, g_errors),
        ),
        xaxis = :log10,
        yaxis = :log10,
        label = label,
    )
end

Plots.pyplot()
Plots.PyPlotBackend()
# pgfplotsx()

# Doing plot() clears the current figure.
Plots.plot(title = "Compute/accuracy tradeoff")
plot_scatter!(euler_bptt_results_fixed, "Euler, BPTT")
plot_scatter!(backsolve_results, "Neural ODE")
plot_scatter!(backsolve_checkpointing_results, "Neural ODE with chkpt.")
plot_scatter!(interp_results, "CTPG (ours)")
Plots.savefig("lqr_tradeoff.pdf")
