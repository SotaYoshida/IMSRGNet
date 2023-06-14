# IMSRGNet
Code and data of IMSRG-Net: arXiv:xxxxx

## To try IMSRG-Net

* `imsrgnet.py`: sample code to run the IMSRG-Net model and to generate operator files (in hd5 fmt used in NuclearToolkit.jl).
* `logfile`: log files of IMSRG-Net results and IMSRG(2) results w/ approximated Magnus operators
* `data_omega_eta_weights`: model parameters and operators used in the paper
* `snts`: NN potential file used in `NuclearToolkit.jl`. Only ones w/ $e_\mathrm{max} \leq 8$ are provided in this repository. For larger $e_\mathrm{max}$, you have to edit `optional_parameters.jl` and run `make_chiEFTint()` function in `NuclearToolkit.jl/chiEFTint.jl`.
* `valencespace`: valence space effective interactions derived IMSRG-Net or VS-IMSRG(2)

You can try IMSRG-Net with the Python script:
```Python
python3 imsrgnet.py
```
and it generates approximated Magnus operators like `ann_omegavec_emax4_emn5002n3n_O16_s106.50.h5`

Note that I do not include all the operator files for larger model space, due to the limitation of the file size.  
If you need those, please do not hesitate contact me or open issues on this repository.

## Author's environment:

I cut a branch of NuclearToolkit.jl [IMSRG-Net-v0](https://github.com/SotaYoshida/NuclearToolkit.jl/tree/IMSRG-Net-v0) to reproduce the results. Note that one may not get *exactly the same* PyTorch results with different environment.
See [Discussions](https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047/13).  

- OS: Ubuntu 20.04.6LTS  
- CPU: i9-10940X
- GPU: NVIDIA RTX A4000
- Python: v3.8.10
- torch: v2.0.0
- Julia: v 1.8.5 
    - NuclearToolkit.jl: IMSRG-Net-v0 branch (=v0.3.4)

## To run NuclearToolkit.jl

* `sample_script_NuclearToolkit.jl`: sample script to run NuclearToolkit.jl to evaluate IMSRG evolved operators from files. See the section "Note on carrying out IMSRG from files" below for more details. This can be used to generate chiral potentials too.
* `optional_parameters.jl`: file to specify some parameters for NuclearToolkit.jl. See the [document](https://sotayoshida.github.io/NuclearToolkit.jl/stable/parameters/) of NuclearToolkit.jl for more details.

If you are new to Julia language, install `Glob` and `NuclearToolkit` through the Julia's package manager:
```julia
using Pkg
Pkg.add("Glob")
Pkg.add("NuclearToolkit")
```
before running Julia scripts.
```bash
julia -t 12 sample_script_NuclearToolkit.jl
```


### Note on carrying out IMSRG flow from files

In `NuclearToolkit.jl` v $\geq$ 0.3.2, one can evaluate any IMSRG evolved operators with a specific Magnus operator from files. The main API is `hf_main` function in `src/hartreefock.jl` and it takes an optional argument `restart_from_files`.
See the sample script `sample_script_NuclearToolkit.jl` in this repository for more details.


# How to cite

I would be so grateful if you would create your own models using this repository and the NuclearToolkit.jl 😉

The IMSRG-Net may in future be integrated into my Julia package [NuclearToolkit.jl](https://github.com/SotaYoshida/NuclearToolkit.jl).  
Please consider to cite the paper for [IMSRG-Net(will be linked)](url) and [NuclearToolkit.jl](https://joss.theoj.org/papers/10.21105/joss.04694) (instead of this repository!). Thank you.
