

using ACEfit, ACEbase
using ACEbase.Testing: println_slim, print_tf

using ACEfit: siterate, titerate 
using Base.Threads

##

@info("Very basic iteration test.")

data = [ Dat(n) for n = 1:100 ]
f_data = [dat.config^2 for dat in data] 

@info("Test siterate")
sf_data = zeros(Int, length(data))
sf = (i, dat) -> (sleep(0.01); sf_data[i] = dat.config^2)
ACEfit.siterate(sf, data)
println_slim(@test sf_data == f_data)

@info("Test titerate")
@info("nthreads == $(nthreads())")
if nthreads() == 1 
   @info("If possible this should be tested with multiple threads...")
end 
tf_data = zeros(Int, length(data))
tf = (i, dat) -> (sleep(0.01); tf_data[i] = dat.config^2)
ACEfit.titerate(tf, data)
println_slim(@test tf_data == f_data)

## 