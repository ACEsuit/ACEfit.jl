# TODO: documentation

abstract type AbstractData end

function count_observations(d::AbstractData) end

function feature_matrix(d::AbstractData) end

function target_vector(d::AbstractData) end

function weight_vector(d::AbstractData) end

function row_info(data)
    row_start = ones(Int,length(data))
    row_count = ones(Int,length(data))
    for (i,d) in enumerate(data)
       row_count[i] = count_observations(d)
       i<length(data) && (row_start[i+1] = row_start[i] + row_count[i])
    end
    return row_start, row_count
end
