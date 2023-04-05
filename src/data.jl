# TODO: documentation

abstract type AbstractData end

function count_observations(d::AbstractData) end

function feature_matrix(d::AbstractData) end

function target_vector(d::AbstractData) end

function weight_vector(d::AbstractData) end
