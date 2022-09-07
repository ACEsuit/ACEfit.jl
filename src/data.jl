# TODO: documentation

abstract type AbstractData end

function countobservations(d::AbstractData) end

function designmatrix(d::AbstractData) end

function targetvector(d::AbstractData) end

function weightvector(d::AbstractData) end
