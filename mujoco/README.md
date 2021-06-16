# neural mujoco
testing neural ode's with mujoco in the loop

FOR VISUALIZATION
one must run julia with
```
$ MESA_LOADER_DRIVER_OVERRIDE=i965 julia --project
```

# startup

```bash
$ cd neuralmujoco/src
$ export JULIA_NUM_THREADS=8 # relevant in the future
$ julia --project -O3
```

Then in the REPL, instantiate the project and dependencies, include the script file that's really not organized.

```julia
julia> ]
(node_test) pkg> instantiate
(node_test) pkg> ctrl-c
julia> include("rch.jl") # reacher
julia> pt = do_train(p0, 200, ADAM(0.01))
julia> rolls, trajs = saverollouts("/tmp/ode_rch.jlso", f, mjd, solver, policy, pt, T, 10);

julia> rch = Reacher()
julia> using LyceumMuJoCoViz
julia> visualize(rch, trajectories=JLSO.load("/tmp/ode_rch.jlso")[:traj])
```

