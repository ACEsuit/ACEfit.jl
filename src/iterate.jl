

# ---------- Single-threaded iteration over configurations 
#  this is just a prototype implementation for tests and compatibility




# ---------- Multi-threaded iteration

import ProgressMeter
using ProgressMeter: Progress

using Base.Threads: @threads, nthreads, SpinLock, threadid, lock, unlock


"""
`tfor(f, rg; verbose=true, msg="tfor", costs = ones(Int, length(rg)))`
Multi-threaded for loop. At each iteration the function f(n) is executed,
where n loops over `rg`.
"""
function tfor(f, rg; verbose=true, msg="tfor", costs = ones(Int, length(rg)),
                     maxnthreads=nthreads())
   p = Progress(sum(costs))
   p_ctr = 0
   t0 = time_ns()
   nthr = max(1, min(nthreads(), maxnthreads))
   if nthr == 1
      verbose && println("$msg in serial")
      dt = verbose ? 1.0 : Inf
      for (n, c) in zip(rg, costs)
         f(n)
         if verbose
            p_ctr += c
            ProgressMeter.update!(p, p_ctr)
         end
      end
   else
      if verbose
         @info("$msg with $(nthreads()) threads")
         ProgressMeter.update!(p, 0)  # not sure this is useful/needed
         p_lock = SpinLock()
      end
      # sort the tasks by cost: do the expensive ones first
      Isort = sortperm(costs, rev=true)
      # remember what the last job was
      last_job = 0
      last_job_lock = SpinLock()
      # start a simple loop over nthreads() just to split this
      # into parallel "tasks"
      rgidx = Vector{Int}(undef, nthreads())
      @threads for i = 1:nthreads()
         while true
            # acquire a new job index
            tid = threadid()
            lock(last_job_lock)
            last_job += 1
            if last_job > length(Isort)
               unlock(last_job_lock)
               break # break the while loop and then wait for the
                     # other threads to finish
            end
            rgidx[tid] = Isort[last_job]
            unlock(last_job_lock)
            # do the job
            f(rg[rgidx[tid]])
            # submit progress
            if verbose
               lock(p_lock)
               p_ctr += costs[rgidx[tid]]
               ProgressMeter.update!(p, p_ctr)
               unlock(p_lock)
            end
         end
      end
   end
   t1 = time_ns()
   verbose && @info("Elapsed: $(round((t1-t0)*1e-9, digits=1))s")
   return nothing
end




# ---------- Distributed iteration 
