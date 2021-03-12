import LinearAlgebra: norm

SPRING_CONNECTIVITY = let
    arr = zeros(n_objects, n_springs)
    for (i, (a, b, _, _, _)) in enumerate(eachrow(springs))
        arr[a+1, i] = -1
        arr[b+1, i] = 1
    end
    arr
end

function julia_forces_fn(x, u)
    # x is (n_objects, 2) and u is (n_springs,)

    # Constants from the outside world.
    spring_length = springs[:, 3]
    stiffness = springs[:, 4]
    actuation = springs[:, 5]

    # (n_springs, 2)
    dist = -SPRING_CONNECTIVITY' * x
    # (n_springs,). For some reason sum(A, dims=2) gives you back a (n, 1) instead of just (n,).
    length = sqrt.(sum(dist .^ 2, dims=2)[:]) .+ 1e-4
    # (n_springs,)
    target_length = spring_length .* (1 .+ actuation .* u)
    # (n_springs, 2). All but the `dist` at the end are (n_springs,)
    impulses = (length - target_length) .* stiffness ./ length .* dist
    acc = SPRING_CONNECTIVITY * impulses

    # Apply gravity.
    hcat(acc[:, 1], acc[:, 2] .- 4.8)
end
