module LinearEnv

function linear_env(floatT, x_dim, A, B, Q, R)
    function dynamics(x, u)
        A * x + B * u
    end

    function cost(x, u)
        x' * Q * x + u' * R * u
    end

    function sample_x0()
        randn(floatT, x_dim)
    end

    dynamics, cost, sample_x0
end

end
