from util.emd_module.emd_module import emdModule
import torch

def earth_mover_distance(input1, input2, eps=0.05, iters=3000):
    emd = emdModule()
    dis, _ = emd(input1.transpose(2, 1), input2.transpose(2, 1), eps, iters)
    return torch.mean(torch.sqrt(dis))