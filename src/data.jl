"""
ACEfit users should define a type of the form:
    UserData <: AbstractData

Several functions acting on such a type should be implemented:
    count_observations
    feature_matrix
    target_vector
    weight_vector
"""
abstract type AbstractData end

"""
Returns the corresponding number of rows in the design matrix.
"""
function count_observations(d::AbstractData) end

"""
Returns the corresponding design matrix (A) entries.
"""
function feature_matrix(d::AbstractData) end

"""
Returns the corresponding target vector (Y) entries.
"""
function target_vector(d::AbstractData) end

"""
Returns the corresponding weight vector (W) entries.
"""
function weight_vector(d::AbstractData) end
