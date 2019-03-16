from __future__ import print_function
import numpy as np
import torch
import torch.optim

from image_prior_depen import *

def optimize(optimizer_type, parameters, closure, LR, num_iter, inpaintinglog):
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
            closure(j, inpaintinglog)
            optimizer.step()
    else:
        assert False
        
        
        
def inpainting(seq_np, mask_np, cuda = False, iterations = 100, inpaintinglog = None):
    
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
    
    num_channels_down = [128] * 3
    num_channels_up =   [128] * 3
    num_channels_skip =  [128] * 3 
    filter_size_up = 3
    filter_size_down = 3
    upsample_mode='nearest'
    filter_skip_size=1
    need_sigmoid=True
    need_bias=True 
    pad=pad
    act_fun='LeakyReLU'


    net = skip(input_depth, seq_np.shape[0], #change skip function in models/skip.py
               num_channels_down = [16, 32, 64, 64, 64],
               num_channels_up = [16, 32, 64, 64, 64],
               num_channels_skip = [16, 32, 64, 64, 64],  
               filter_size_up = 3, filter_size_down = 3, 
               upsample_mode = 'nearest', filter_skip_size = 1,
               need_sigmoid = True, need_bias = True, pad = pad, act_fun = act_fun).type(dtype)
    

    net = net.type(dtype) 
    net_input = get_noise(input_depth, INPUT, seq_np.shape[1]).type(dtype) #tensor 

    
    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)
    
    if inpaintinglog != None:
        inpaintinglog.add_net_parameters([NET_TYPE, pad, OPT_OVER, OPTIMIZER, INPUT, 
                                     input_depth, LR, reg_noise_std, num_iter, cuda, s,
                                        num_channels_down,
                                       num_channels_up,
                                       num_channels_skip,  
                                       filter_size_up, filter_size_down, 
                                       upsample_mode, filter_skip_size,
                                       need_sigmoid, need_bias, pad, act_fun])

        inpaintinglog.init_log()


    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_var = np_to_torch(seq_np).type(dtype)
    mask_var = np_to_torch(mask_np).type(dtype)
    
    def closure(i, inpaintinglog):
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
        
        if inpaintinglog != None:
            inpaintinglog.loss.append(total_loss)
            if i % inpaintinglog.out_nps_every == 0:
                out_np = torch_to_np(out)
                inpaintinglog.compare_log(i, out_np)

        return total_loss
    
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    p = get_params(OPT_OVER, net, net_input) # list of tensors to optimize over !! in optimize
    
    start_time = time.time()
    optimize(OPTIMIZER, p, closure, LR, num_iter, inpaintinglog) # optimize is in utils/common.utils
    elapsed_time = time.time() - start_time
    print("\ntime: {}s".format(elapsed_time))
    
    out_np = torch_to_np(net(net_input))
    inpaintinglog.end_log()
    
    return out_np