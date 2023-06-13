import h5py
import glob
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
from pytorch_memlab import profile
import seaborn as sns
sns.set_style()
from sys import byteorder
from struct import unpack, pack
import sys

def write_fvec_h5(x, y, emax, nuc, refvec, org_dim, idxs_hdf5, omegaeta="omega",addlabel=""):
    fn = "ann_"+omegaeta+"vec_emax"+str(emax)+"_"+inttype+"_"+nuc+"_s"+str("%6.2f" % x).strip()+addlabel+".h5"
    f = h5py.File(fn, 'w')
    s = x
    E0 = 0.0 # y[0] removed 03/02
    if skip_connection:
        y += refvec
    f.create_dataset('s', data = s)
    f.create_dataset('Es', data = E0)
    f.create_dataset('dim', data = org_dim)
    i,j = idxs_hdf5["n1b"]; f.create_dataset('n1b', data = y[i:j])
    i,j = idxs_hdf5["p1b"]; f.create_dataset('p1b', data = y[i:j])
    i,j = idxs_hdf5["pp2b"]; f.create_dataset('pp2b', data = y[i:j])
    i,j = idxs_hdf5["pn2b"]; f.create_dataset('pn2b', data = y[i:j])
    i,j = idxs_hdf5["nn2b"]; f.create_dataset('nn2b', data = y[i:j])
    f.close()

def read_operator_hdf5(fns):
    x_vec = [ ]
    y_vec = [ ]
    idxs_hdf5 = {}
    for fn in fns:
        f = h5py.File(fn, 'r')
        s = f["s"][()]
        E0 = f["Es"][()]
        dim = f["dim"][()]
        vec_p1b = f["p1b"][()]
        vec_n1b = f["n1b"][()]
        vec_pp2b = f["pp2b"][()]
        vec_pn2b = f["pn2b"][()]
        vec_nn2b = f["nn2b"][()]
        y = list(vec_p1b) + list(vec_n1b) + list(vec_pp2b) + list(vec_pn2b) + list(vec_nn2b)
        if len(y) != dim:
            print("dim",dim, "!=","len(y)",len(y))
            exit()
        x_vec += [ s ]
        y_vec += [ y ]
    idxs_hdf5["p1b"] = [0,len(vec_p1b)]
    idxs_hdf5["n1b"] = [len(vec_p1b),len(vec_p1b)+len(vec_n1b)]; idx_i = len(vec_p1b)+len(vec_n1b)
    idxs_hdf5["pp2b"] = [idx_i,idx_i+len(vec_pp2b)]; idx_i = idx_i+len(vec_pp2b)
    idxs_hdf5["pn2b"] = [idx_i,idx_i+len(vec_pn2b)]; idx_i = idx_i+len(vec_pn2b)
    idxs_hdf5["nn2b"] = [idx_i,idx_i+len(vec_nn2b)]

    x_vec = np.array(x_vec).reshape((len(fns),1))
    y_vec = np.array(y_vec).reshape((len(fns),dim)) 
    return x_vec, y_vec, idxs_hdf5

def get_omega_bins(dirname,pid,nuc):
    fs = glob.glob(dirname+"/omega_vec_"+str(pid)+nuc+"_*"+".h5")
    nums = sorted([ float(f.split("_s")[-1].split(".h5")[0]) for f in fs])
    fns = [ dirname+"/omega_vec_"+str(pid)+nuc+"_s"+str("%6.2f" % num).strip()+".h5" for num in nums ]
    return fns

def get_eta_bins(dirname,pid,nuc):
    fs = glob.glob(dirname+"/eta_vec_"+str(pid)+nuc+"_*"+".h5")
    nums = sorted([ float(f.split("_s")[-1].split(".h5")[0]) for f in fs])
    fns = [ dirname+"/eta_vec_"+str(pid)+nuc+"_s"+str("%6.2f" % num).strip()+".h5" for num in nums ]
    return fns

class DataSet:
    def __init__(self,emax,dirname,pid,nuc,nummax,TF_normalize,ratio=1.0):
        self.emax = emax
        self.dirname = dirname
        self.pid = pid
        self.fns_Omegas = get_omega_bins(dirname,pid,nuc)
        self.fns_Etas = get_eta_bins(dirname,pid,nuc)
        self.TF_normalize = TF_normalize
        self.num_data = len(self.fns_Omegas)
        self.ntrain = min(int(ratio*self.num_data),nummax)
        svals = [ float(fn.split("_s")[-1].split(".h5")[0]) for fn in self.fns_Omegas]
        if valencespace:
            for idx,sval in enumerate(svals):
                if sval > 50.0:
		            idx_conv = idx
                    break
            self.sconv = svals[idx_conv]
	        print("x giving converged free-space IMSRG calc.",self.sconv)

        self.idxs_test  = [ idx for idx in range(self.num_data-1,self.num_data)]
        idx_s20 = svals.index(20.0)
        self.idxs_train = [ idx for idx in range(idx_s20-nummax_train+1,idx_s20+1,train_mod) ]
        self.idxs_valid = [ idx for idx in range(idx_s20-nummax_train+1-num_valid,idx_s20-nummax_train+1,train_mod)]
        
    def read_OperatorJL(self, fns_O, fns_E, load_eta=True):
        x, y,idxs_hdf5 = read_operator_hdf5(fns_O)
        xdummy, yd, dumm = read_operator_hdf5(fns_E)
        return x, y, yd, idxs_hdf5

    def load_data(self,tol_omega=0.0):
        fns_O_train = [self.fns_Omegas[idx] for idx in self.idxs_train]
        fns_E_train = [self.fns_Etas[idx] for idx in self.idxs_train]

        x_train, y_train, yd_train,idxs_hdf5 = self.read_OperatorJL(fns_O_train,fns_E_train)
        y_mean = np.mean(y_train)
        y_std = max(np.std(y_train),1.e-3) 
        ymax = np.max(y_train)
        ymin = np.min(y_train)
        
        fns_O_valid = [self.fns_Omegas[idx] for idx in self.idxs_valid]
        fns_E_valid = [self.fns_Etas[idx] for idx in self.idxs_valid]
        x_valid, y_valid, yd_valid, dum = self.read_OperatorJL(fns_O_valid,fns_E_valid)

        fns_O_test = [self.fns_Omegas[idx] for idx in self.idxs_test]
        fns_E_test = [self.fns_Etas[idx] for idx in self.idxs_test]
        x_test, y_test, yd_test, dum = self.read_OperatorJL(fns_O_test,fns_E_test)

        ### for valence flow
        if valencespace:
            x_train -= self.sconv
            x_valid -= self.sconv
            x_test  -= self.sconv
        
        org_dim = len(y_train[-1,:])
        zeroidx_eta = []
        non0idx_eta = []
        yd = yd_train[-1,:]

        s20idx = list(x_train.reshape(1,-1)[0,:]).index(20.0)
        y = y_train[s20idx,:]
        refvec = copy.deepcopy(y)
        if skip_connection:
            y_train -= refvec
            y_test -= refvec
            y_valid -= refvec
        hit = hit0 = 0
        for idx,tmp in enumerate(y):
            if tr_idx != -1:
                if idx >= tr_idx:
                    continue
            if yd[idx] != 0.0:
                non0idx_eta += [idx]
            else:
                zeroidx_eta += [idx]
        if self.TF_normalize:
            normalize(y_train, y_mean, y_std)
            normalize(yd_train, y_mean, y_std)
            normalize(y_test, y_mean, y_std)
            normalize(yd_test, y_mean, y_std)
            normalize(y_valid, y_mean, y_std)

        x_train = torch.tensor(x_train,dtype=torch.float32)
        y_train = torch.tensor(y_train,dtype=torch.float32)
        yd_train = torch.tensor(yd_train,dtype=torch.float32)
        x_valid = torch.tensor(x_valid,dtype=torch.float32)
        y_valid = torch.tensor(y_valid,dtype=torch.float32)
        x_test = torch.tensor(x_test,dtype=torch.float32)
        y_test = torch.tensor(y_test,dtype=torch.float32)
        yd_test = torch.tensor(yd_test,dtype=torch.float32)
        return x_train, y_train, yd_train, x_valid,y_valid, x_test, y_test, yd_test, y_mean, y_std, ymax,ymin,zeroidx_eta,non0idx_eta, org_dim, refvec, idxs_hdf5

def get_nonzero_idxs(vec):
    idxs = [ ]
    for (i,v) in enumerate(vec):
        if v !=0.0:
            idxs += [ i ]
    return idxs

def normalize(target, vmean, vstd, rev=False,replace=False):
    if rev:
        target *= vstd
    else:
        target /= vstd
    if replace:
        return target

class PINN(nn.Module):
    def __init__(self, input_dim, num_node_hlayer,num_hlayer, num_node_edge, output_dim, activation_function):
        super(PINN, self).__init__()
        self.biasTF = bias_TF
        self.input_dim = input_dim
        self.num_node_hlayer = num_node_hlayer
        self.num_hlayer = num_hlayer
        self.output_dim = output_dim
        self.num_node_edge = num_node_edge
        self.num_node_1 = num_node_hlayer
        self.fc_inp = nn.Linear(self.input_dim, self.num_node_1,bias=self.biasTF)
        self.act_inp = nn.Tanh()
        self.hl1 = nn.Linear(self.num_node_1,self.num_node_hlayer,bias=self.biasTF)
        self.act1 = nn.Softplus()
        self.hl_edge = nn.Linear(self.num_node_hlayer,self.num_node_edge,bias=self.biasTF)
        self.act_edge = nn.Softplus()
        self.fc_out = nn.Linear(self.num_node_edge, self.output_dim, bias=edge_bias)

    def forward(self, x):
        out = self.act_inp(self.fc_inp(x))
        out = self.act1(self.hl1(out))
        out = self.act_edge(self.hl_edge(out))
        out = self.fc_out(out)
        return out

    def __call__(self, inp):
        return self.forward(inp)

    def loss_mse(self, x, y,ynn):
        return nn.MSELoss()(ynn,y)

    def loss_eta(self,yd,idxs, mode="all", diffmode="c"):
        if diffmode == "f":
            yp = self(x_train_plus[idxs,:])
            ydiff = (yp-ynn) / eps
        elif diffmode == "c":   
            yp = self(x_train_plus[idxs,:])
            ym = self(x_train_minus[idxs,:])
            ydiff = (yp-ym) / (2*eps)
        elif diffmode == "b":
            ym = self(x_train_minus[idxs,:])
            ydiff = (ynn-ym) / eps
        if mode == "zero":
            losseta = lam_der_0 * nn.MSELoss()(ydiff[:,zeroidx_eta],yd[:,zeroidx_eta])
        elif mode == "nonzero":
            losseta = lam_der * nn.MSELoss()(ydiff[:,non0idx_eta],yd[:,non0idx_eta])
        elif mode == "weighted":
            losseta = lam_der_0 * nn.MSELoss()(ydiff[:,zeroidx_eta],yd[:,zeroidx_eta]) + lam_der * nn.MSELoss()(ydiff[:,non0idx_eta],yd[:,non0idx_eta])
        elif mode == "all":
            losseta = lam_der * nn.MSELoss()(ydiff,yd)
        return losseta

    def loss_eta_ad(self,x,y):
        dim = y.shape[-1]
        Osum = torch.sum(torch.abs(y))
        dOsum_ds = torch.autograd.grad(Osum, x, grad_outputs=torch.ones_like(Osum), create_graph=True)[0][0]
        loss = (dOsum_ds **2) / (x.shape[0]*dim)
        return loss.item()

    def w2norm(self,ret,verbose=False):
        norm_sum = 0.0
        str1 = str2 = ""
        arr = [epoch]
        for key in keys:
            w = self.state_dict()[key]
            tnorm = torch.norm(w)
            arr += [torch.norm(w).item()]
            if "weight" in key:
                str1 += str(key).ljust(15)+" "+str("%9.3e" % tnorm.item())+"   "
            if "bias" in key:
                str2 += str(key).ljust(15)+" "+str("%9.3e" % tnorm.item())+"   "
            norm_sum += torch.norm(w).item()
        if verbose:
            print(str1)
            print(str2)
        ret += [arr]
        return norm_sum

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

def closure():
    optimizer.zero_grad()
    ynn = model(x)
    myloss = nn.MSELoss()(y,ynn)
    idxs = list(range(num_train_data))
    myloss += model.loss_eta(yd,idxs)
    myloss.backward()
    return myloss

def estimate_res_size(input_dim, num_node_hlayer, num_hlayers, num_train):
    w_size =  4* ( input_dim * num_node_hlayer + num_hlayers*num_node_hlayer**2   ) / (1024*1024*1024)
    d_size = num_train * 4 * input_dim *2 / (1024*1024*1024)
    print("size of weight parameters (GB):", w_size)
    print("size of data (GB):", d_size)

def select_optimizer(model, learning_rate, optimizer_type):
    using_LBFGS = False
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1 ,momentum=0.9,weight_decay=wd) 
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=wd)
    elif optimizer_type == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr = learning_rate, weight_decay=wd)
    elif optimizer_type == "L-BFGS":
        optimizer = torch.optim.LBFGS(model.parameters())
        using_LBFGS = True
    else:
        print("unsupported optimizer_type",optimizer_type)
        exit()
    return optimizer, using_LBFGS

def batch_loader(num_train_data,num_batch):
    perm = np.random.choice(num_train_data,num_train_data,replace=False)
    idxs = [ [] for i in range(num_batch)]
    for n,idx in enumerate(perm):
        target = int(np.floor(n / (num_train_data/num_batch)))
        idxs[target] += [ idx ]
    return idxs

def print_vec(lab,obj):
    cstr = '{:<20}'.format(lab)
    for tmp in obj:
        cstr += str("%9.4e" % tmp)+" "    
    print(cstr)


if __name__ == '__main__' :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tr_idx = -1
    save_weights = "F" # set "T" when you want to train IMSRG-Net
    load_weights = "T"

    nummax_train = 10
    num_valid = 5
    train_mod = 1
    
    TF_normalize = True
    bias_TF = True
    edge_bias = False
    skip_connection = True
    finetuning = False 
    opt_der = True  
    retrain = False
    valencespace = False
        
    model_type = "PINN"
    num_batch = nummax_train // train_mod 
    optimizer_type = "AdamW"; org_lr = 1.e-2; org_wd = 0.1
    activation_function = "Softplus"

    org_lam_der = 1.e+2
    org_wd = 0.1    
    lam_der_0 = 0.0
    
    num_hlayer = 1
    num_node_hlayer = 48
    num_node_edge = 16
    twostep = False; org_lr = 3.e-3
   
    num_epochs = 5000
    num_epoch_show = 500

    if retrain :
        num_epochs = 50
        num_epoch_show = num_epochs//10
        save_weights = "F"
        load_weights = "T"
        optimizer_type="AdamW"
        learning_rate = 1.e-3

    is_save_weights = True if save_weights == "T" else False
    is_load_weights = True if load_weights == "T" else False

    for emax in [4]: #for emax in [4,6,8,10]:
        for inttype in ["em500","emn5002n3n"]:
            for nuc in ["O16","Ca40"]:
                seed = 1234
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                #torch.backends.cudnn.deterministic = True
                #torch.use_deterministic_algorithms = True
                #torch.backends.cudnn.benchmark = False
                lam_der = org_lam_der
                learning_rate = org_lr
                wd = org_wd
                
                dirname = os.path.expanduser("data_omega_eta_weights/runlog_e"+str(emax)+"_"+inttype)   
                fname = dirname+"/omega_vec_*"+nuc+"*.h5"
                fns = glob.glob(fname)
                numdata = len(fns)
                nummax = numdata - 1
                if numdata==0:
                    print(fname, "not found"); exit()

                pid = int( fns[0].split("vec_")[-1].split(nuc)[0])

                mydata = DataSet(emax,dirname,pid,nuc,nummax,TF_normalize)
                
                x_train,y_train,yd_train,x_valid,y_valid,x_test,y_test,yd_test,y_mean,y_std,ymax,ymin,zeroidx_eta,non0idx_eta,org_dim,refvec,idxs_hdf5 = mydata.load_data()
                sumx = torch.sum(x_train,dim=0).item()

                input_dim = 1
                output_dim = y_train.shape[1]
                num_train_data = y_train.shape[0]
                num_test_data = y_test.shape[0]
                model_path = dirname+'/model_e'+str(emax)+"hl"+str(num_hlayer)+"node"+str(num_node_hlayer)+"_"+str(num_node_edge)+nuc+'.pth'
                model = PINN(input_dim,num_node_hlayer,num_hlayer,num_node_edge,output_dim,activation_function).to(device)
                keys = list(model.state_dict().keys())
                optimizer,using_LBFGS = select_optimizer(model, learning_rate, optimizer_type)
                
                print("---------------------------------")
                print("x train",list(x_train.cpu().detach().numpy().reshape(1,-1)[0,:]),
                      "test",list(x_test.cpu().detach().numpy().reshape(1,-1)[0,:]))
                print("x valid",list(x_valid.cpu().detach().numpy().reshape(1,-1)[0,:]))                
                print("device:",device, "nuc",nuc, "emax", emax, "inttype", inttype)
                print("pid",pid,"dim input:", input_dim, "output", output_dim,
                      " # train ", num_train_data, "batchsize", num_batch, " # test ", num_test_data)
                print("Normalize",TF_normalize, " y_mean/std ",str("%12.4e" % y_mean), str("%12.4e" % y_std),
                      "ymax/ymin",str("%12.4e" % ymax),str("%12.4e" % ymin))
                print("skip connection",skip_connection,"Optimize w/ derivative ",opt_der)
                print("optimizer:",optimizer_type, "learning rate:", learning_rate,
                    "weight_decay", wd,"lam_der",lam_der,"lam_der_0",lam_der_0)
                print("model",model)
                print("---------------------------------")

                lossdata = [ ]
                wdata = [ ]
                eps = 1.e-3 
                xp = np.array([ x[0].item()+eps for x in x_train]).reshape(num_train_data,-1)
                xm = np.array([ x[0].item()-eps for x in x_train]).reshape(num_train_data,-1)
                x_train_plus  = torch.tensor(xp,dtype=torch.float32,device=device)
                x_train_minus = torch.tensor(xm,dtype=torch.float32,device=device)

                if is_load_weights:
                    if len(glob.glob(model_path)) ==0:
                        print(model_path ,"not found!")
                    else:
                        model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
                        print(model_path, " loaded!")

                best_loss = 1.e+100
                best_epoch = -1
                if is_save_weights or ( is_load_weights==is_save_weights==False) or retrain:
                    model.train()
                    for epoch in range(num_epochs+1):
                        idxs = batch_loader(num_train_data,num_batch)
                        loss_sum_epoch = loss_mse_epoch = loss_der_epoch = loss_eta_epoch = loss_test_epoch = 0.0
 
                        for batch in range(num_batch):
                            batch_idxs = idxs[batch]
                            x = x_train[batch_idxs,:].to(device)
                            y = y_train[batch_idxs,:].to(device)
                            ynn = model(x)
                            yd = yd_train[batch_idxs,:].to(device)
                            optimizer.zero_grad(set_to_none=True)

                            loss = model.loss_mse(x,y,ynn)
                            loss_mse_epoch += loss.item() * y_std**2
                            loss_eta = model.loss_eta(yd,batch_idxs)        
                            loss_eta_epoch += loss_eta.item() * y_std**2
                            if opt_der:
                                loss += loss_eta
                            loss_sum_epoch += loss.item() * y_std**2

                            loss.backward()
                            optimizer.step()
                            del loss, loss_eta,ynn,yd
                        l2 = 0.0
                        for w in model.parameters():
                            l2 = l2 + torch.norm(w).item()**2
                        l2 *= 1.e-6

                        xvald = x_valid.to(device)
                        yvald = y_valid.to(device)
                        yvNN = model(xvald)
                        loss_valid_epoch = nn.MSELoss()(yvNN,yvald).item() * y_std**2
                    
                        if epoch % num_epoch_show == 0 or epoch == num_epochs:
                            xtest = x_test.to(device)                            
                            ytest = y_test.to(device)
                            yNNtest = model(xtest)
                            testloss = nn.MSELoss()(yNNtest,ytest).item() * y_std**2
                            print("Epoch", str("%8i" % epoch)+"/"+str("%8i" % num_epochs),
                                  "loss",str("%9.3e" % (loss_sum_epoch+loss_valid_epoch)),"loss_mse", str("%9.3e" % (loss_mse_epoch)),
                                  "loss_eta", str("%9.3e" % (loss_eta_epoch)),
                                  "LossTest",str("%9.3e" % (testloss)),"(",str("%9.3e" % (testloss/y_std**2)),")",
                                  "weight", str("%9.3e" % l2))                        
                        if epoch % 10 == 0 :
                            xtest = x_test.to(device)
                            ytest = y_test.to(device)
                            yNNtest = model(xtest)
                            testloss = nn.MSELoss()(yNNtest,ytest).item() * y_std**2
                            loss_test_epoch = testloss
                            model.w2norm(wdata)
                            lossdata += [ [epoch,loss_mse_epoch,loss_eta_epoch,loss_valid_epoch,loss_test_epoch,l2] ]
                        if loss_sum_epoch + loss_valid_epoch < best_loss and epoch > round(0.8*num_epochs):
                            if abs(loss_sum_epoch -(loss_mse_epoch + loss_eta_epoch)) > 1.e-6:
                                print("diff sum", loss_sum_epoch, " mse+eta ", loss_mse_epoch+loss_eta_epoch)
                            best_epoch = epoch
                            best_loss  = loss_sum_epoch + loss_valid_epoch
                            torch.save(model.state_dict(), model_path)

                model.eval()
                if emax < 10:
                    x = x_train.to(device)
                    y = y_train.to(device)
                    yd = yd_train.to(device)
                    xtest  = x_test.to(device)
                    ytest  = y_test.to(device)
                    xvald = x_valid.to(device)
                    yvald = y_valid.to(device)
                    ynn = model(x)
                    idxs = range(num_train_data)
                    loss_mse = model.loss_mse(x,y,ynn).item()* y_std**2
                    loss_eta = model.loss_eta(yd,idxs).item()* y_std**2
                    loss = loss_mse + loss_eta 
                    yNNtest = model(xtest)
                    testloss = nn.MSELoss()(yNNtest,ytest).item()* y_std**2
                    yvNN = model(xvald)
                    loss_valid_epoch =  nn.MSELoss()(yvNN,yvald).item() * y_std**2
                    print("Best: ", best_epoch,"loss",str("%9.3e" % (loss)),
                          " LossValid", str("%9.3e" % (loss_valid_epoch)),
                          " LossTest",str("%9.3e" % (testloss)))
                else:
                    print("Best: ", best_epoch,"bestloss",str("%9.3e" % (best_loss)))
                    
                print("test:")
                x = x_test.to(device)
                y  = y_test.to(device)
                yd = yd_test.to(device)
                yNN = model(x)
                for n in range(num_test_data):
                    loss = nn.MSELoss()(yNN[n],y[n])
                    x = x_test[n,:].to(device)
                    tyd = yd[n,:]
                    idx_0 = [ ]
                    idx_non0 = [ ]
                    print("x",str("%9.2f" %(x[0].item())),
                          " loss",str("%9.3e" % (loss.item() * y_std**2)),
                          "(",str("%9.3e" % (loss.item())),")")
                    yout = yNN[n,:].cpu().detach().numpy()
                    if TF_normalize:
                        yout = normalize(yout, y_mean, y_std,True,True)
                    
                    if n >= num_test_data-10:
                        write_fvec_h5(x[0].item(), yout, emax, nuc, refvec, org_dim, idxs_hdf5, "omega")
                        
                del model
                del x_train,y_train,yd_train,x_valid,y_valid
                del x_test,y_test,yd_test,zeroidx_eta,non0idx_eta,org_dim,refvec,idxs_hdf5
