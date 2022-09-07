# TODO: documentation

abstract type AbstractData end

function countobservations(d::AbstractData) end

function designmatrix(d::AbstractData) end

function targetvector(d::AbstractData) end

function weightvector(d::AbstractData) end

function row_info(data)
    row_start = ones(Int,length(data))
    row_count = ones(Int,length(data))
    for (i,d) in enumerate(data)
       row_count[i] = countobservations(d)
       i<length(data) && (row_start[i+1] = row_start[i] + row_count[i])
    end
    return row_start, row_count
end
