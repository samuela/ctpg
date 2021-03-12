# Specifying Array types here is important for ReverseDiff.jl to be able to hit the tracked version of this function
# below. Why is this the case? Who knows. Because x and v are actually ReshapedArrays, it ends up being easier to
# dispatch on u. See https://github.com/JuliaDiff/ReverseDiff.jl/issues/150#issuecomment-692649803.
function difftaichi_forces_fn(x, u::Array)
    mass_spring.forces_fn(np.array(x, dtype=np.float32), np.array(u, dtype=np.float32))
end
function difftaichi_forces_fn(x, u::ReverseDiff.TrackedArray)
    ReverseDiff.track(difftaichi_forces_fn, x, u)
end
ReverseDiff.@grad function difftaichi_forces_fn(x, u)
    # We reuse these for both the forward and backward.
    x_np = np.array(x, dtype=np.float32)
    u_np = np.array(u, dtype=np.float32)
    function pullback(ΔΩ)
        mass_spring.forces_fn_vjp(x_np, u_np, np.array(ΔΩ, dtype=np.float32))
    end
    mass_spring.forces_fn(x_np, u_np), pullback
end
Zygote.@adjoint function difftaichi_forces_fn(x, u)
    # We reuse these for both the forward and backward.
    x_np = np.array(x, dtype=np.float32)
    u_np = np.array(u, dtype=np.float32)
    function pullback(ΔΩ)
        mass_spring.forces_fn_vjp(x_np, u_np, np.array(ΔΩ, dtype=np.float32))
    end
    mass_spring.forces_fn(x_np, u_np), pullback
end
