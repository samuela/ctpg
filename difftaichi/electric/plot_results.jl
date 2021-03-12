import JLSO
import ThreadPools: qmap
import Plots
import Statistics: median, quantile, mean
import PyCall: pyimport

np = pyimport("numpy")

results_dir = "results/2020-10-28T14:45:12.999-6c49e0ed4bb8fd7a6cb1c1b819eff9369d0a670f-electric-many"

# Parallel version doesn't give any speedup.
# @time results = qmap(1:32) do i
#     JLSO.load(joinpath(results_dir, "seed$i/results.jlso"))
# end
@time results = [JLSO.load(joinpath(results_dir, "seed$i/results.jlso")) for i in 1:32]

# (n_iter, n_seeds)
ppg_losses_per_iter = hcat([res[:ppg_results].loss_per_iter for res in results]...)
bptt_losses_per_iter = hcat([res[:bptt_results].loss_per_iter for res in results]...)

# See https://stackoverflow.com/questions/59562325/moving-average-in-julia
moving_average(vs, n = 64) = [mean(@view vs[max(1, i - n + 1):i]) for i in 1:length(vs)]

# We show the (errorbar, 1 - errorbar) percentiles as error bars.
errorbar = 0.05

Plots.pyplot()
Plots.PyPlotBackend()

# scalefontsizes mutates the current default size, so that sucks...
Plots.resetfontsizes()
Plots.scalefontsizes(2)

# per iteration
ppg_losses_per_iter = hcat([moving_average(col) for col in eachcol(ppg_losses_per_iter)]...)
bptt_losses_per_iter = hcat([moving_average(col) for col in eachcol(bptt_losses_per_iter)]...)

every = 1000

# TODO: fix the ribbon on these.
Plots.plot(title = "DiffTaichi electric experiment")
ys = median(bptt_losses_per_iter[1:every:end, :], dims = 2)
Plots.plot!(
    1:every:size(bptt_losses_per_iter, 1),
    ys,
    ribbon = (
        ys - [quantile(bptt_losses_per_iter[ix, :], errorbar) for ix in 1:every:size(bptt_losses_per_iter, 1)],
        -ys + [quantile(bptt_losses_per_iter[ix, :], 1 - errorbar) for ix in 1:every:size(bptt_losses_per_iter, 1)]
    ),
    xlabel = "Iteration",
    ylabel = "Loss",
    label = "DiffTaichi (BPTT)"
)
ys = median(ppg_losses_per_iter[1:every:end, :], dims = 2)
Plots.plot!(
    1:every:size(ppg_losses_per_iter, 1),
    ys,
    ribbon = (
        ys - [quantile(ppg_losses_per_iter[ix, :], errorbar) for ix in 1:every:size(ppg_losses_per_iter, 1)],
        -ys + [quantile(ppg_losses_per_iter[ix, :], 1 - errorbar) for ix in 1:every:size(ppg_losses_per_iter, 1)]
    ),
    label = "CTPG (ours)"
)
Plots.savefig("poop_per_iter.pdf")

# per nf
nf_eval = range(1, 1.5e8, length = 100)
ppg_losses_per_nf = hcat([begin
    r = res[:ppg_results]
    nfs = cumsum(r.nf_per_iter + r.n∇ₓf_per_iter + r.n∇ᵤf_per_iter)
    np.interp(nf_eval, nfs, r.loss_per_iter |> moving_average)
end for res in results]...)
bptt_losses_per_nf = hcat([begin
    r = res[:bptt_results]
    nfs = cumsum(r.nf_per_iter + r.n∇ₓf_per_iter + r.n∇ᵤf_per_iter)
    np.interp(nf_eval, nfs, r.loss_per_iter |> moving_average)
end for res in results]...)

# Plots.plot(title = "DiffTaichi electric experiment")
Plots.plot()
ys = median(bptt_losses_per_nf, dims = 2)
Plots.plot!(
    nf_eval,
    ys,
    ribbon = (
        ys - [quantile(row, errorbar) for row in eachrow(bptt_losses_per_nf)],
        -ys + [quantile(row, 1 - errorbar) for row in eachrow(bptt_losses_per_nf)],
    ),
    xlabel = "Number of function evaluations",
    ylabel = "Loss",
    label = "DiffTaichi (BPTT)",
)
ys = median(ppg_losses_per_nf, dims = 2)
Plots.plot!(
    nf_eval,
    ys,
    ribbon = (
        ys - [quantile(row, errorbar) for row in eachrow(ppg_losses_per_nf)],
        -ys + [quantile(row, 1 - errorbar) for row in eachrow(ppg_losses_per_nf)],
    ),
    label = "CTPG (ours)"
)
Plots.ylims!((0.0, 0.5))
Plots.savefig("poop_per_nf.pdf")

# per wallclock
# time_eval = 1:1e6:1e7
time_eval = range(1, 2.25e4, length=100)
ppg_losses_per_time = hcat([begin
    r = res[:ppg_results]
    time = cumsum(r.elapsed_per_iter) / 1e9
    np.interp(time_eval, time, r.loss_per_iter |> moving_average)
end for res in results]...)
bptt_losses_per_time = hcat([begin
    r = res[:bptt_results]
    time = cumsum(r.elapsed_per_iter) / 1e9
    np.interp(time_eval, time, r.loss_per_iter |> moving_average)
end for res in results]...)

# Plots.plot(title = "DiffTaichi electric experiment")
Plots.plot()
ys = median(bptt_losses_per_time, dims = 2)
Plots.plot!(
    time_eval,
    ys,
    ribbon = (
        ys - [quantile(row, errorbar) for row in eachrow(bptt_losses_per_time)],
        -ys + [quantile(row, 1 - errorbar) for row in eachrow(bptt_losses_per_time)],
    ),
    xlabel = "Wallclock time (s)",
    ylabel = "Loss",
    label = "DiffTaichi (BPTT)",
)
ys = median(ppg_losses_per_time, dims = 2)
Plots.plot!(
    time_eval,
    ys,
    ribbon = (
        ys - [quantile(row, errorbar) for row in eachrow(ppg_losses_per_time)],
        -ys + [quantile(row, 1 - errorbar) for row in eachrow(ppg_losses_per_time)],
    ),
    label = "CTPG (ours)"
)
Plots.ylims!((0.0, 0.5))
Plots.savefig("poop_per_time.pdf")
