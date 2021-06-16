# Copyright 2017 The dm_control Authors.
# Copyright (c) 2019 Colin Summers, The Contributors of LyceumMuJoCo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
struct Tolerance{T<:Real,F}
    bounds::Tuple{T,T}
    margin::T
    sigmoid::F
    value_at_margin::T
    function Tolerance{T}(;bounds = (zero(T), zero(T)),
                          margin=zero(T),
                          sigmoid=quadratic,
                          value_at_margin=T(0.1)) where T
        lo, up = bounds
        lo <= up || throw(DomainError("Lower bound must be <= upper bound"))
        margin >= 0 || throw(DomainError("`margin` must be > 0"))

        new{T, typeof(sigmoid)}(T.(bounds),
                                T(margin),
                                sigmoid,
                                T(value_at_margin))
    end
end

function (t::Tolerance{T})(x::Real) where T
    #x = T(x)
    lo, up = t.bounds
    in_bounds = lo <= x <= up
    if iszero(t.margin)
        return in_bounds ? one(x) : zero(x)
    else
        d = (x < lo ? lo - x : x - up) / t.margin
        value = in_bounds ? one(x) : t.sigmoid(d, t.value_at_margin)
        return convert(typeof(x), value)
    end
end

#= shouldnt need this
function (t::Tolerance)(xs::AbstractArray)
    t.(xs)
end
=#

const _DEFAULT_VALUE_AT_MARGIN = 0.1

function tolerance(
    x;
    bounds = (0, 0),
    margin = 0,
    sigmoid = quadratic,
    value_at_margin = _DEFAULT_VALUE_AT_MARGIN,
)
    lo, up = bounds
    lo <= up || throw(DomainError("Lower bound must be <= upper bound"))
    margin >= 0 || throw(DomainError("`margin` must be > 0"))
    _tolerance(x, lo, up, margin, sigmoid, value_at_margin)
end

function _tolerance(x::Real, lo, up, margin, sigmoid, vmargin)
    in_bounds = lo <= x <= up
    if iszero(margin)
        return ifelse(in_bounds, one(x), zero(x))
    else
        d = ifelse(x < lo, lo - x, x - up) / margin
        value = in_bounds ? one(x) :  sigmoid(d, vmargin)
        return convert(typeof(x), value)
    end
end

function _tolerance(xs::AbstractArray, args...)
    map(x -> _tolerance(x, args...), xs)
end

function linear(x::T, value_at_1::T)::T where T
    scale = one(x) - value_at_1
    scaled_x = x * scale
    abs(scaled_x) < one(x) ? one(x) - scaled_x : zero(x)
end

function quadratic(x::T, value_at_1::T)::T where T
    scale = sqrt(one(x) - value_at_1)
    scaled_x = x * scale
    abs(scaled_x) < one(x) ? one(x) - scaled_x ^ 2 : zero(x)
end
#quadratic(xs::AbstractArray, value_at_1::Number) = map(x -> quadratic(x, value_at_1), xs)
#quadratic!(xs::AbstractArray, value_at_1::Number) = xs .= quadratic.(xs, value_at_1)


function _check_value_at_1(x::Number)
    0 < x < 1 || throw(DomainError("`value_at_1` must be in range [0, 1)"))
end
function _check_value_at_1_strict(x::Number)
    0 < x < 1 || throw(DomainError("`value_at_1` must be in range (0, 1)"))
end
