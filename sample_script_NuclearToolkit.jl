using NuclearToolkit
using Glob
using Printf

function imsrg_emulator()
    # make chiral potentials
    # make_chiEFTint(); exit()

    # IMSRG part
    valencespace = false 
    @assert !valencespace "to carry out VS-IMSRG calculation, you need to prepare Ω(s) files"

    hw = 20
    dirsnt = "snts"
    nucs_f = ["O16","Ca40"]
    nucs_v = ["O18","Cr48"]
    nucs = ifelse(valencespace,nucs_v,nucs_f)
    for emax in [4] # [4,6,8,10]    
        for nuc in nucs
            corenuc = ifelse(nuc=="O18","O16","Ca40")
            vspace = ifelse(nuc=="O18","sd-shell","pf-shell")
            for sntf in ["tbme_em500n3lo_srg2.0hw$(hw)emax$(emax).snt.bin",
                         "tbme_emn500n4lo_2n3n_srg2.0hw$(hw)emax$(emax).snt.bin"]
                sntf = dirsnt*"/"*sntf
                println("#############################")
                println("sntf $sntf")
                tf_em = occursin("em500",sntf)                
                inttype = ifelse(tf_em,"em500","emn5002n3n")

                if !valencespace
                    # IMSRG emulator 
                    fns = glob("ann_omegavec_emax$(emax)_$(inttype)_$(nuc)_s*.h5")
                    svals = sort([parse(Float64,split(split(fn,"_s")[end],".h5")[1]) for fn in fns])
                    fns_conv = [ "ann_omegavec_emax$(emax)_$(inttype)_$(nuc)_s$(strip(@sprintf("%8.2f",sval))).h5" for sval in svals]
                    hf_main([nuc],sntf,hw,emax;doIMSRG=true,
                            restart_from_files=[fns_conv,[]],fn_params="ann_sample_params.jl",Operators=["Rp2"])
                else
                    # VS-IMSRG emulator
                    ## specifyng the omegavec file giving conveged IMSRG(2) calculation
                    dirname = "./data_omega_eta_weights/vsrunlog_e$(emax)_$(inttype)/"
                    fns = glob(dirname * "omega_vec_*$(nuc)_s*.h5")
                    svals = sort([parse(Float64,split(split(fn,"_s")[end],".h5")[1]) for fn in fns])
                    sconv = 0
                    for sval in svals
                        if sval > 50.0
                            sconv = sval
                            break
                        end
                    end
                    fn_conv = glob(dirname * "omega_vec_*$(nuc)_s$(strip(@sprintf("%8.2f",sconv))).h5")
                    ## specifyng the omegavec file for VS-IMSRG(2) flow, i.e. output of (VS-)IMSRG-Net
                    fn_vs = "ann_omegavec_emax$(emax)_$(inttype)_$(nuc)_s"*strip(@sprintf("%8.2f",svals[end]))*".h5"

                    hf_main([nuc],sntf,hw,emax;verbose=false,doIMSRG=true,
                            valencespace = vspace,
                            ref = "core",
                            corenuc = corenuc,
                            restart_from_files=[[fn_conv],[fn_vs]],
                            fn_params="ann_sample_params.jl")
                end                
                println("\n\n")
            end
        end
    end   
end
imsrg_emulator()

