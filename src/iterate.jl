import ProgressMeter
using ProgressMeter: Progress


# TODO: maybe data cannot always be held in memory and would be better 
# to have the options of either retrieving from memory or from disk
# could introduce a `retrieve` function which dispatches depending on how 
# the list of training data is passed in.
# 
# TODO: flag to turn off the progressmeter? Should this be a flag separate from 
# the verbose flag? 

# ---------- Single-threaded iteration over configurations 
#  this is just a prototype implementation for tests and compatibility


function siterate(f, data; verbose=true, msg = "serial", costs = cost.(data))
   progmtr = Progress(sum(costs))
   progctr = 0 

   t0 = time_ns()

   for (idat, (dat, c)) in enumerate(zip(data, costs))
      f(idat, dat)
      progctr += c 
      ProgressMeter.update!(progmtr, progctr)
   end

   t1 = time_ns()
   verbose && @info("Elapsed: $(round((t1-t0)*1e-9, digits=1))s")

   return nothing 
end


## ---------- Multi-threaded iteration
#
#
#using Base.Threads: @threads, nthreads, SpinLock, threadid, lock, unlock
#
#
#"""
#`titerate(f, data; kwargs...)`
#Multi-threaded map loop. At each iteration the function f(dat) is executed,
#where `dat in data`. The order is not necessarily preserved. In fact the 
#`costs` array is used to sort the data by decreasing cost to ensure the 
#most costly configurations are encountered first. This helps avoid threads 
#without work at the end of the loop.
#"""
#function titerate(f, data; verbose=true, msg="multi-threaded", 
#                           costs = cost.(data),
#                           maxnthreads=nthreads() )
#   progmtr = Progress(sum(costs))
#   progctr = 0
#   t0 = time_ns()
#   nthr = max(1, min(nthreads(), maxnthreads))
#   if verbose
#      @info("Iterate with $(nthreads()) threads")
#      ProgressMeter.update!(progmtr, 0)  # not sure this is useful/needed
#      prog_lock = SpinLock()
#   end
#   # sort the tasks by cost: do the expensive ones first
#   Isort = sortperm(costs, rev=true)
#   # remember what the last job was
#   last_job = 0
#   last_job_lock = SpinLock()
#   # start a simple loop over nthreads() just to split this
#   # into parallel "tasks"
#   rgidx = Vector{Int}(undef, nthreads())
#   @threads for i = 1:nthreads()
#      while true
#         # acquire a new job index
#         tid = threadid()
#         lock(last_job_lock)
#         last_job += 1
#         if last_job > length(Isort)
#            unlock(last_job_lock)
#            break # break the while loop and then wait for the
#                  # other threads to finish
#         end
#         rgidx[tid] = Isort[last_job]
#         unlock(last_job_lock)
#         # do the job
#         f(rgidx[tid], data[rgidx[tid]])
#         # submit progress
#         if verbose
#            lock(prog_lock)
#            progctr += costs[rgidx[tid]]
#            ProgressMeter.update!(progmtr, progctr)
#            unlock(prog_lock)
#         end
#      end
#   end
#   t1 = time_ns()
#   verbose && @info("Elapsed: $(round((t1-t0)*1e-9, digits=1))s")
#   return nothing
#end



# ---------- Distributed iteration 

# TODO: Andres? 
