"""
This file is for Gym cartpole environment https://github.com/openai/gym/tree/master/gym/envs/classic_control.
These are based on https://github.com/WilsonWangTHU/mbbl
"""

using LyceumBase
using Shapes
using Base: @propagate_inbounds

struct CartpoleGym{D,SP,OP,AP,RP} <: AbstractEnvironment
    state::D
    obs::D
    action::D
    time::D
    nextobs::D
    ctrlrange::D
    
    sp::SP
    op::OP
    ap::AP
    rp::RP
    function CartpoleGym{T}() where T<:AbstractFloat
        nobs = 4
        naction = 1

        ctrlrange = zeros(T, 2)
        ctrlrange[1] = -1.
        ctrlrange[2] = 1.
       
        sp = VectorShape(T, nobs)
        op = VectorShape(T, nobs)
        ap = VectorShape(T, naction)
        rp = ScalarShape(T)
        s = zeros(T, nobs)
        o = zeros(T, nobs)
        a = zeros(T, naction)
        env = new{typeof(s), 
        typeof(sp), typeof(op), typeof(ap), typeof(rp)}(s,o,a,zeros(T,1),
                                                        zeros(T, nobs),
                                                        ctrlrange,
                                                        sp,op,ap,rp)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{CartpoleGym}, n::Integer, T=Float64)
    Tuple(CartpoleGym{T}() for _ in 1:n)
end
CartpoleGym() = first(tconstruct(CartpoleGym, 1))

@inline LyceumBase.statespace(env::CartpoleGym) = env.sp
@inline LyceumBase.obsspace(env::CartpoleGym) = env.op
@inline LyceumBase.actionspace(env::CartpoleGym) = env.ap
@inline LyceumBase.rewardspace(env::CartpoleGym) = env.rp


@inline function LyceumBase.getstate!(state, env::CartpoleGym) 
    #state .= env.state
    state[1] = env.state[1] # unroll for speed
    state[2] = env.state[2]
    state[3] = env.state[3]
    state[4] = env.state[4]
end
@inline function LyceumBase.setstate!(env::CartpoleGym, state) 
    #env.state .= state
    env.state[1] = state[1] # unroll for speed
    env.state[2] = state[2]
    env.state[3] = state[3]
    env.state[4] = state[4]
end

@inline LyceumBase.getaction!(action, env::CartpoleGym) = action .= env.action
@inline LyceumBase.setaction!(env::CartpoleGym, action) = env.action .= action

@propagate_inbounds function LyceumBase.isdone(state, action, obs, env::CartpoleGym)
    false
end
@propagate_inbounds function LyceumBase.getobs!(obs, env::CartpoleGym)
    getstate!(env.obs, env) # abuse of functions but valid
    getstate!(obs, env)
end
@propagate_inbounds function LyceumBase.getreward(s, a, o, env::CartpoleGym) 
    x = o[1]
    theta = o[3]
    up_reward = cos(theta)
    distance_penalty_reward = -0.01 * (x^2)
    return up_reward + distance_penalty_reward
end
@propagate_inbounds function LyceumBase.geteval(s, a, o, env::CartpoleGym)
    x = o[1]
    theta = o[3]
    up_reward = cos(theta)
    distance_penalty_reward = -0.01 * (x^2)
    return up_reward + distance_penalty_reward
end
@propagate_inbounds function LyceumBase.reset!(env::CartpoleGym)
    randreset!(env)
end
@propagate_inbounds function LyceumBase.randreset!(env::CartpoleGym)
    env.state .= 0.
    env.state .= rand(Uniform(-0.05, 0.05), length(env.state))
    getobs!(env.obs, env)
    env.time[1] = 0.0
    env
end

@propagate_inbounds function LyceumBase.step!(env::CartpoleGym)
    
    state, act, ctrlrange = env.state, env.action, env.ctrlrange
    x, x_dot, theta, theta_dot = state

    # In Benchmark paper  action set to 1 if >0 0 otherwise
    # In Gym it becomes self.force_mag = 10.0 if 1 -10. otherwise
    force = 0.
    if act[1] > 0.; force = 10.; else; force = -10.; end;
    sintheta, costheta = sincos(theta)
    
    # self.polemass_length = 0.1 * 0.5 = 0.05 self.total_mass=1.0+0.1=1.1
    # self.gravity = 9.8
    temp     = (force + 0.05 * theta_dot^2 * sintheta) / 1.1
    thetaacc = (9.8 * sintheta - costheta * temp) / (0.5 * (4.0 / 3.0 - 0.1 * costheta^2 / 1.1))
    xacc     = temp - 0.05 * thetaacc * costheta / 1.1

    # self.tau = 0.02   EULER default
    x     = x + 0.02 * x_dot;          x_dot     = x_dot + 0.02 * xacc
    theta = theta + 0.02 * theta_dot;  theta_dot = theta_dot + 0.02 * thetaacc
        
    env.state[1] = x;  env.state[2] = x_dot;  env.state[3] = theta;  env.state[4] = theta_dot
    env.time[1] += timestep(env)
    env
end

@inline LyceumBase.timestep(env::CartpoleGym) = 1.0
@inline Base.time(env::CartpoleGym) = env.time[1]

#199.85387503411363
