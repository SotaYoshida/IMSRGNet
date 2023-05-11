# IMSRGNet
Code and data of IMSRG-Net: arXiv:xxxxx

* imsrgnet.py: sample code to train the IMSRG-Net and to make operator files (in hd5 fmt used in NuclearToolkit.jl).
* runlog: model parameters and operators used in the paper

I do not include all the operator files for larger model space (emax > 6), due to the limitation of the file size.  
If you need those, please do not hesitate contact me or open issues on this repository.


## Author's environment:

I cut a branch of NuclearToolkit.jl [IMSRG-Net-v0]() to reproduce the resutls.
Note that one cannot *exactly* reproduce PyTorch results with different environment.
See [Discussions](https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047/13).  
The following info is just for your reference.

- OS: Ubuntu 20.04.6LTS  
- CPU: i9-10940X
- GPU: NVIDIA RTX A4000
- Python: v3.8.10
- torch: v2.0.0
- Julia: v 1.8.5 


# How to cite

I would be so grateful if you would create your own models using this repository and the NuclearToolkit.jl 😉

The IMSRG-Net may in future be integrated into my Julia package [NuclearToolkit.jl](https://github.com/SotaYoshida/NuclearToolkit.jl).  
Please consider to cite the paper for [IMSRG-Net](url) and [NuclearToolkit.jl](https://joss.theoj.org/papers/10.21105/joss.04694) (not this repository!)
Thank you.
