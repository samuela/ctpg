import Hyperopt: @hyperopt, Hyperband
import Optim: optimize
import Random: seed!
import Plots: plot

seed!(0)

loss(x) = sum(x.^2)

ho = @hyperopt for i in 0, sampler in Hyperband(R=5000)
    if state === nothing
        state = randn(500)
    end

    res = optimize(loss, state, iterations=i)

    # return the minimum value and a state from which to continue the optimization
    res.minimum, res.minimizer
end

# `ho.history` is the returned `res.minimizer` for each run in order. Because
# hyperband pauses and restarts runs, these will be in order of their occurrence
# in the hyperband algorithm. `ho.results` will be the corresponding function
# values.

plot(accumulate(min, ho.results))
