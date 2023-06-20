using ACEfit
using PythonCall
using Documenter

DocMeta.setdocmeta!(ACEfit, :DocTestSetup, :(using ACEfit); recursive = true)

makedocs(;
         modules = [ACEfit],
         authors = "Christoph Ortner <christophortner0@gmail.com> and contributors",
         repo = "https://github.com/ACEsuit/ACEfit.jl/blob/{commit}{path}#{line}",
         sitename = "ACEfit.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://ACEsuit.github.io/ACEfit.jl",
                                  assets = String[]),
         pages = [
             "Home" => "index.md",
         ])

deploydocs(;
           repo = "github.com/ACEsuit/ACEfit.jl",
           devbranch = "main")
