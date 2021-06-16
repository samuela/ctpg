using LyceumMuJoCo
using UnsafeArrays
using Random
using Shapes
#------------------Constructing Environment-------------------#


struct Hopper{Sim, OSpace} <: AbstractMuJoCoEnvironment
    sim::Sim
    #statespace::SSpace
    obsspace::OSpace
    randreset_distribution::Uniform{Float64}
    function Hopper(sim::MJSim)
        ospace = MultiShape(
            cropped_qpos = VectorShape(Float64, sim.m.nq - 1),
            #t = ScalarShape(Float64),
            #qpos = VectorShape(Float64, sim.m.nq),
            qvel = VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim), typeof(ospace)}(
            sim, ospace, Uniform(-0.5, 0.5)
        )
        reset!(env)
    end
end

Hopper() = first(tconstruct(Hopper, 1))

function LyceumBase.tconstruct(::Type{Hopper}, n::Integer)
    modelpath = joinpath(@__DIR__, "hopper.xml")
    return Tuple(Hopper(s) for s in tconstruct(MJSim, n, modelpath, skip=4))
end
function ensembleconstruct(::Type{Hopper}, n::Integer)
    modelpath = joinpath(@__DIR__, "hopper.xml")
    return Tuple(Hopper(MJSim(modelpath, skip = 4)) for m=1:n )
end

@inline LyceumMuJoCo.getsim(env::Hopper) = env.sim
@inline LyceumMuJoCo.obsspace(env::Hopper) = env.obsspace

@propagate_inbounds function LyceumMuJoCo.getobs!(obs, env::Hopper)
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    qvel = env.sim.d.qvel
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        copyto!(shaped.cropped_qpos, qpos[2:end])
        #shaped.t = env.sim.d.time
        #copyto!(shaped.qpos, qpos)
        #shaped.cropped_qpos[2] = mod(shaped.cropped_qpos[2] + pi, 2pi) - pi
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10.0, 10.0)
        #shaped.qvel .= clamp.(qvel, -10.0, 10.0) #./ 10.0
        #shaped.qvel .= qvel
    end
    obs
end


@propagate_inbounds function LyceumMuJoCo.getreward(state, action, obs, env::Hopper)
    os = obsspace(env)(obs)
    #reward = os.qvel[1] #* 10.0
    reward = 0.0
    reward -= 3. * (os.cropped_qpos[1] - 1.3)^2
    reward -= 0.1 * norm(action)^2
    reward += 1.0
end
@propagate_inbounds function LyceumMuJoCo.geteval(state, action, obs, env::Hopper)
    #obs[6] * 10.0
    #state[1] # x position
    state[2] + 1.25 # z height with offset
end

@propagate_inbounds function LyceumMuJoCo.reset!(env::Hopper)
    reset!(env.sim)
    env
end

@propagate_inbounds function LyceumMuJoCo.randreset!(rng::AbstractRNG, env::Hopper)
    LyceumMuJoCo.reset_nofwd!(env.sim)
    m, d = env.sim.m, env.sim.d
    #perturb!(rng, env.randreset_distribution, d.qpos)
    mag = 0.1
    d.qpos[3] = mag * rand(rng, Uniform(-pi, pi))
    for i=4:m.nq # latter joints
        d.qpos[i] = mag * rand(rng, Uniform(m.jnt_range[1,i], m.jnt_range[2,i]))
    end
    d.qpos[2] = 1.25 # z/height stuff

    perturb!(rng, env.randreset_distribution, d.qvel)
    d.qvel .*= 0.1 # slow it down a little with larger distribution
    forward!(env.sim)
    env
end

@propagate_inbounds function LyceumMuJoCo.isdone(state, obs, env::Hopper)
    #return obs[1] < -1. || obs[6] < 3.
    return false
end

@inline _torso_x(shapedstate::ShapedView, ::Hopper) = shapedstate.simstate.qpos[1]
@inline _torso_x(env::Hopper) = env.sim.d.qpos[1]
@inline _torso_height(shapedstate::ShapedView, ::Hopper) = shapedstate.simstate.qpos[2]
@inline _torso_ang(shapedstate::ShapedView, ::Hopper) = shapedstate.simstate.qpos[3]
