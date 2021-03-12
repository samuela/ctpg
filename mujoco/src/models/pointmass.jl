
using Distances 

struct PointMassV2{S <: MJSim, O} <: AbstractMuJoCoEnvironment
    sim::S
    obsspace::O
    function PointMassV2(sim::MJSim)
        obsspace = MultiShape(
            agent_xy_pos  = VectorShape(Float64, 2),
            agent_xy_vel  = VectorShape(Float64, 2),
            target_xy_pos = VectorShape(Float64, 2)
        )
        new{typeof(sim), typeof(obsspace)}(sim, obsspace)
    end
end

function LyceumBase.tconstruct(::Type{PointMassV2}, N::Integer)
    modelpath = joinpath(@__DIR__, "pointmass.xml")
    Tuple(PointMassV2(s) for s in tconstruct(MJSim, N, modelpath, skip=1))
end

PointMassV2() = first(tconstruct(PointMassV2, 1))



@inline LyceumMuJoCo.getsim(env::PointMassV2) = env.sim
@inline LyceumMuJoCo.obsspace(env::PointMassV2) = env.obsspace

function getobsfromstate(state, env::PointMassV2)
    m = env.sim.m
end

@propagate_inbounds function LyceumMuJoCo.getobs!(obs, env::PointMassV2)
    #@boundscheck checkaxes(obsspace(env), obs)
    dn = env.sim.dn
    shaped = obsspace(env)(obs)
    @uviews shaped @inbounds begin
        shaped.agent_xy_pos .= dn.xpos[:x, :agent], dn.xpos[:y, :agent]
        shaped.agent_xy_vel .= dn.qvel[:agent_x], dn.qvel[:agent_y]
        shaped.target_xy_pos .= dn.xpos[:x, :target], dn.xpos[:y, :target]
    end
    obs
end


@propagate_inbounds function LyceumMuJoCo.getreward(::Any, ::Any, obs, env::PointMassV2)
    #@boundscheck checkaxes(obsspace(env), obs)
    shaped = obsspace(env)(obs)
    reward = 1.0
    @uviews shaped @inbounds begin
        reward -= euclidean(shaped.agent_xy_pos, shaped.target_xy_pos)
    end
    #exp(reward)
    #exp(log(10)/0.5 * time(env)) * (reward)
    #reward
    #0.9999^(time(env)/timestep(env)) * reward # doesn't overshoot; something something value is critically damped?
    #1.0001^(time(env)/timestep(env)) * reward # doesn't overshoot; something something value is critically damped?
    reward
end

function runningcost(state, action, obs, env::PointMassV2)
    #-1e3 * norm(action)^2
    0
end
function terminalcost(state, action, obs, env::PointMassV2)
    getreward(state, action, obs, env)
end



@propagate_inbounds function LyceumMuJoCo.geteval(::Any, ::Any, obs, env::PointMassV2)
    #@boundscheck checkaxes(obsspace(env), obs)
    shaped = obsspace(env)(obs)
    @uviews shaped @inbounds begin
        euclidean(shaped.agent_xy_pos, shaped.target_xy_pos)
    end
end


@propagate_inbounds function LyceumMuJoCo.randreset!(rng::Random.AbstractRNG, env::PointMassV2)
    LyceumMuJoCo.reset_nofwd!(env.sim)
    @inbounds begin
        env.sim.dn.qpos[:agent_x] = rand(rng) * 2.0 - 1.0
        env.sim.dn.qpos[:agent_y] = rand(rng) * 2.0 - 1.0
        env.sim.dn.qpos[:target_x] = 0.0 #rand(rng) * 2.0 - 1.0
        env.sim.dn.qpos[:target_y] = 0.0 #rand(rng) * 2.0 - 1.0
        env.sim.d.qvel[1] = rand(rng) * 2.0 - 1.0
        env.sim.d.qvel[2] = rand(rng) * 2.0 - 1.0
    end
    forward!(env.sim)
    env
end
