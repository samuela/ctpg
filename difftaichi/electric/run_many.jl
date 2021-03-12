"""Script to run many electric jobs in parallel.

We do it this way instead of using Julia's inter-process parallel tools because PyCall only supports one interpreter,
and we don't want to have every job competing over the global interpreter lock.
"""

import Dates

# Child processes inherit ENV, so we use this to limit their threading.
ENV["JULIA_NUM_THREADS"] = 1

experiment_dir = "results/$(Dates.now())-electric-many"
mkdir(experiment_dir)
@info "Outputting results to $experiment_dir"

num_random_seeds = 32

# See https://discourse.julialang.org/t/dir-doesnt-work-properly-within-backtick-notation/48566.
dir = @__DIR__
jobs = [
    pipeline(
        `julia --project $dir/run.jl --experiment_dir=$experiment_dir/seed$i --rng_seed=$i`,
        stdout="$experiment_dir/seed$(i)_stdout.txt",
        stderr="$experiment_dir/seed$(i)_stderr.txt")
    for i in 1:num_random_seeds
]

@info "Running $num_random_seeds electric jobs in parallel..."
# We can't run all at once due to memory pressure (running on n64), so split jobs into chunks to run in parallel.
for chunk in Iterators.partition(jobs, 16)
    print(".")
    # & runs commands in parallel. See https://docs.julialang.org/en/v1/manual/running-external-programs/.
    run(reduce(&, chunk))
end
