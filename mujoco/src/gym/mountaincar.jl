"""
This file is for Gym mountaincar environment https://github.com/openai/gym/tree/master/gym/envs/classic_control.
These are based on https://github.com/WilsonWangTHU/mbbl
"""

using LyceumBase
using Shapes
using Distributions
using Base: @propagate_inbounds

struct MountainCar{D,SP,OP,AP,RP} <: AbstractEnvironment
    state::Vector{D}
    obs::Vector{D}
    action::Vector{D}
    time::Vector{D}
    nextobs::Vector{D}
    
    ctrlrange::Vector{D}
    maxspeed::D
    power::D
    posrange::Vector{D}
    
    sp::SP
    op::OP
    ap::AP
    rp::RP
    function MountainCar{T}() where T<:AbstractFloat
        nobs = 2
        naction = 1

        ctrlrange    = zeros(T, 2)
        ctrlrange[1] = -1.;  ctrlrange[2] = 1.
        maxspeed     = T(0.07); power        = T(0.0015)
        posrange     = zeros(T, 2)
        posrange[1]  = -1.2; posrange[2]  = 0.6

        sp = VectorShape(T, nobs)
        op = VectorShape(T, nobs)
        ap = VectorShape(T, naction)
        rp = ScalarShape(T)
        s = zeros(T, nobs)
        o = zeros(T, nobs)
        a = zeros(T, naction)
        env = new{T, 
        typeof(sp), typeof(op), typeof(ap), typeof(rp)}(s,o,a,zeros(T,1),
                                                        zeros(T, nobs),
                                                        ctrlrange, maxspeed, power, posrange,
                                                        sp,op,ap,rp)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{MountainCar}, n::Integer)
    Tuple(MountainCar{Float64}() for _ in 1:n)
end
MountainCar() = first(tconstruct(MountainCar, 1))

@inline LyceumBase.statespace(env::MountainCar) = env.sp
@inline LyceumBase.obsspace(env::MountainCar) = env.op
@inline LyceumBase.actionspace(env::MountainCar) = env.ap
@inline LyceumBase.rewardspace(env::MountainCar) = env.rp

@propagate_inbounds @inline function LyceumBase.getstate!(state, env::MountainCar) 
    #copyto!(state, env.state)
    state[1] = env.state[1] # unroll for speed
    state[2] = env.state[2]
end
@propagate_inbounds @inline function LyceumBase.setstate!(env::MountainCar, state) 
    #env.state .= state
    env.state[1] = state[1] # unroll for speed
    env.state[2] = state[2]
end

@inline LyceumBase.getaction!(action, env::MountainCar) = action .= env.action
@inline LyceumBase.setaction!(env::MountainCar, action) = env.action .= action

@propagate_inbounds function LyceumBase.isdone(state, action, obs, env::MountainCar)
    false
end
@propagate_inbounds function LyceumBase.getobs!(obs, env::MountainCar)
    #copyto!(env.obs, env.state)
    #copyto!(obs, env.state)
    env.obs[1] = env.state[1] # unroll for speed
    env.obs[2] = env.state[2]
    obs[1] = env.state[1]
    obs[2] = env.state[2]
end
@propagate_inbounds function LyceumBase.getreward(s, a, o, env::MountainCar) 
    # In benchmark paper
    return o[1]
end
@propagate_inbounds function LyceumBase.geteval(s, a, o, env::MountainCar)
    return o[1]
end
@propagate_inbounds function LyceumBase.reset!(env::MountainCar)
    randreset!(env)
end
@propagate_inbounds function LyceumBase.randreset!(env::MountainCar)
    env.state[1] = rand(Uniform(-0.6, -0.4))
    env.state[2] = 0.0
    getobs!(env.obs, env)
    env.time[1] = 0.0
    env
end

@propagate_inbounds function LyceumBase.step!(env::MountainCar)
    
    state, act, ctrlrange, maxspeed, power, posrange = env.state, env.action, env.ctrlrange, env.maxspeed, env.power, env.posrange
    
    position = state[1]
    velocity = state[2]
    """
        Note this is from Benchmark paper, not original in Gym env
        corresponding to action = np.clip(action, -1., 1.)
        also in continuous MC Gym env force = min(max(action[0], self.min_action), self.max_action) 
        where min_action -1 max_action +1
    """
    force = clamp(act[1], ctrlrange[1], ctrlrange[2])

    velocity += force * power - 0.0025 * cos(3.0 * position)
    velocity = clamp(velocity, -maxspeed, maxspeed)

    position += velocity
    position = clamp(position, posrange[1], posrange[2])
    if (position == posrange[1]) & (velocity < 0.0);  velocity = 0.0;  end

    env.state[1] = position
    env.state[2] = velocity
    env.time[1] += timestep(env)
    env
end

@inline LyceumBase.timestep(env::MountainCar) = 1.0
@inline Base.time(env::MountainCar) = env.time[1]
