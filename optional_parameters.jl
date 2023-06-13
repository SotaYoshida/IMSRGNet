###--- ChiralEFT ---
n_mesh = 50
emax = 4
#chi_order = 4; pottype="emn500n4lo";calc_3N = true
chi_order = 3; pottype="em500n3lo";calc_3N = false
calc_NN = true

hw = 20.0
srg_lambda = 2.0
tbme_fmt = "snt.bin"

### --- IMSRG ---
smax =  500.0
dsmax = 0.5
denominatorDelta=0.0
BetaCM = 0.0
magnusamethod="NS"
