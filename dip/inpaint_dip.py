from __future__ import print_function
import numpy as np
import torch
import torch.optim
from deep_image_prior_depen import *
import argparse
import os
import datetime
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Generates folder with np array for seq and mask for sequence file')

parser.add_argument( "-p", "--path", required=True, type=str, 
                    help='folder with sqeunces to inpaint')
parser.add_argument( "-n", "--iter", default = 501, type=int, 
                    help='number of iterations')
parser.add_argument( "-e", "--every", default = 500, type=int, 
                    help='save out_np every')
parser.add_argument( "-c", "--cuda", action='store_true', 
                   help = "add for cuda gpu")

args = parser.parse_args()

path = args.path
every = args.every   


if os.path.isdir(path):
    print("got directory {}".format(path))
else:
    print("bad directory")
    exit()
               
class Logger():
    def __init__(self, path, every):
        self.loss = []
        self.every = every
        self.path = path
    def add_net_parameters(self, p):
        net_keys = ["NET_TYPE", "pad", "OPT_OVER", "OPTIMIZER", "INPUT", "input_depth", "LR", "reg_noise_std",
                         "num_iter", "cuda", "num_parameters", 
                       "num_channels_down",
                       "num_channels_up",
                       "num_channels_skip",  
                       "filter_size_up", "filter_size_down", 
                       "upsample_mode", "filter_skip_size",
                       "need_sigmoid", "need_bias", "pad", "act_fun"]

        net_parameters = {net_keys[i]:p[i] for i in range(len(net_keys))}
        num_iter = p[8]
        net_description = ",".join([str(x) for x in p])
        file = open(os.path.join(self.path, "info.txt"), "+a")
        file.write(net_description + "\n")
        file.close()
        
    
    def middle_log(self, i, total_loss, out_torch):
        self.loss.append(total_loss)
        out_np = torch_to_np(out_torch)
        if i % self.every == 0:
            np.save(os.path.join(self.path, "{:05d}_out_np.npy".format(i)), out_np)

    def end_log(self, net, time = None):
        loss = np.array((self.loss))
        np.save(os.path.join(self.path, "loss.npy"), loss)
        torch.save(net.state_dict(), os.path.join(self.path, "net_dict.pty"))
        torch.save(net, os.path.join(self.path, "net.pty"))
        file = open(os.path.join(self.path, "info.txt"), "+a")
        file.write("{:.3f}s\t{}\n".format(time, torch.cuda.get_device_name(0)))
        file.close()



def optimize(optimizer_type, parameters, closure, LR, num_iter, logger):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam' #no change
        parameters: list of Tensors to optimize over ## 
        closure: function, that returns loss variable #no change
        LR: learning rate #no change
        num_iter: number of iterations  #no change
    """
    if optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        for j in range(num_iter):
            optimizer.zero_grad()
#             closure()
            closure(j, logger)
            optimizer.step()
    else:
        assert False
        
        
        
def inpainting(seq_np, mask_np, cuda, iterations, logger):
    
#     seq_np = container.seq_np
#     mask_np = container.mask_np

    if cuda: 
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark =True
        dtype = torch.cuda.FloatTensor
    else:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        dtype = torch.FloatTensor

#     PLOT = False
#     imsize = -1
#     dim_div_by = 64

    NET_TYPE = 'skip_depth6'
    pad = 'reflection' # 'zero'
    OPT_OVER = 'net'
    OPTIMIZER = 'adam'

    INPUT = 'noise'
    input_depth = 32
    LR = 0.01 
    num_iter = iterations
    param_noise = False
    show_every = 5
    figsize = 5 #????
    reg_noise_std = 0.03
    
    num_channels_down = [16] * 5
    num_channels_up =   [16] * 5
    num_channels_skip =  [16] * 5 
    filter_size_up = 3
    filter_size_down = 3
    upsample_mode='nearest'
    filter_skip_size=1
    need_sigmoid=True
    need_bias=True 
    pad=pad
    act_fun='LeakyReLU'


    net = skip(input_depth, seq_np.shape[0], #change skip function in models/skip.py
               num_channels_down = num_channels_down,
               num_channels_up = num_channels_up,
               num_channels_skip = num_channels_skip,  
               filter_size_up = filter_size_up, filter_size_down = filter_size_down, 
               upsample_mode = upsample_mode, filter_skip_size = filter_skip_size,
               need_sigmoid = True, need_bias = True, pad = pad, act_fun = act_fun).type(dtype)
    

    net = net.type(dtype) 
    net_input = get_noise(input_depth, INPUT, seq_np.shape[1]).type(dtype) #tensor 

    
    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)
    

    logger.add_net_parameters([NET_TYPE, pad, OPT_OVER, OPTIMIZER, INPUT, 
                                     input_depth, LR, reg_noise_std, num_iter, cuda, s,
                                        num_channels_down,
                                       num_channels_up,
                                       num_channels_skip,  
                                       filter_size_up, filter_size_down, 
                                       upsample_mode, filter_skip_size,
                                       need_sigmoid, need_bias, pad, act_fun])



    # Loss
   #mse = torch.nn.MSELoss().type(dtype)
    mse = torch.nn.CrossEntropyLoss().type(dtype)
    

    img_var = np_to_torch(seq_np).type(dtype)
    mask_var = np_to_torch(mask_np).type(dtype)
    
    def closure(i, logger):
    #     if param_noise:
    #         for n in [x for x in net.parameters() if len(x.size()) == 4]:
    #             n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        out = net(net_input)
  
        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()
        print ('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
        

        logger.middle_log(i, total_loss, out)
        return total_loss
    
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    p = get_params(OPT_OVER, net, net_input) # list of tensors to optimize over !! in optimize
    
    start_time = time.time()
    optimize(OPTIMIZER, p, closure, LR, num_iter, logger) # optimize is in utils/common.utils
    elapsed_time = time.time() - start_time
    print("\ntime: {}s".format(elapsed_time))
    
    
    logger.end_log(net, elapsed_time)
    
    out_np = torch_to_np(net(net_input))
    return out_np


seq_np = np.load(os.path.join(path, "seq_np.npy"))
mask_np = np.load(os.path.join(path, "mask_np.npy"))
logger = Logger(path, every)
inpainting(seq_np, mask_np, args.cuda, args.iter, logger)
