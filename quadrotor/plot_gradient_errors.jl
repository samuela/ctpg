import JLSO
import LinearAlgebra: norm
import Statistics: median, mean, std
import Plots

data = JLSO.load("213aadd_diffdrive_gradient_error_results.jlso")
backsolve_results = data[:backsolve_results]
backsolve_checkpointing_results = data[:backsolve_checkpointing_results]
interp_results = data[:interp_results]
euler_bptt_results = data[:euler_bptt_results]
gold_standard_results = data[:gold_standard_results]

function plot_scatter!(results, label)
    nf_calls = [sol.nf + sol.n∇f for sol in results]
    g_errors = [
        norm(gold.g - est.g)
        for (gold, est) in zip(gold_standard_results, results)
    ]

    Plots.scatter!(
        nf_calls,
        g_errors,
        xlabel = "Function evaluations",
        ylabel = "L2 error in the gradient",
        xaxis = :log10,
        yaxis = :log10,
        alpha = 0.5,
        markerstrokewidth = 0,
        label = label,
    )
end

Plots.pyplot()
Plots.PyPlotBackend()
# pgfplotsx()

# Doing plot() clears the current figure.
Plots.plot(title = "Compute/accuracy tradeoff")
plot_scatter!(euler_bptt_results, "Euler, BPTT")
plot_scatter!(backsolve_results, "Neural ODE")
plot_scatter!(backsolve_checkpointing_results, "Neural ODE with checkpointing")
plot_scatter!(interp_results, "Ours")
Plots.savefig("diffdrive_tradeoff.pdf")
