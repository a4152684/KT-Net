from tqdm import tqdm
import torch
from dataset import get_dataloader
from agent import get_agent
from util.utils import cycle, ensure_dir
import random
from common import get_config

random.seed(1856)

def writePoints(filename, points_input):
    with open(filename, "w") as f:
        for index in range(points_input.shape[0]):
            x,y,z = points_input[index]
            f.write(str(x)+" "+str(y)+" "+str(z)+ "\n")

def Test(config):
    print("Testing....")

    config.batch_size = 1

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint
    tr_agent.load_ckpt(config)
    #tr_agent.save_ckpt()
    tr_agent.eval()

    test_loader, _ = get_dataloader('test', config)
    num_test = len(test_loader)
    print("total number of test samples: {}".format(num_test))
    test_loader = cycle(test_loader)

    save_dir = 'result/'
    ensure_dir(save_dir)

    # run
    for i in tqdm(range(num_test)):
        data = next(test_loader)

        with torch.no_grad():
            tr_agent.forward(data)
    
    with open(save_dir + "experiment_" + config.dataset_name + ".txt", "a") as f:
        cd_t, cd_p = tr_agent.cd_t_show.detach().cpu().numpy()/tr_agent.item_num, tr_agent.cd_p_show.detach().cpu().numpy()/tr_agent.item_num
        f.write(config.category + ", cd_t: " + str(10000*cd_t) + ", cd_p: " + str(100*cd_p) + "\n")
        print(config.category + ", cd_t: " + str(10000*cd_t) + ", cd_p: " + str(100*cd_p))

if __name__ == "__main__":
    config = get_config('test')
    Test(config)