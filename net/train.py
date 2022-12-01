from collections import OrderedDict
from tqdm import tqdm
from dataset import get_dataloader
from common import get_config
from agent import get_agent

import random
random.seed(1214)

def main():
    # create experiment config containing all hyperparameters
    config = get_config('train')

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    if config.cont:
        tr_agent.load_ckpt(config)

    # create dataloader
    train_loader, val_dataset = get_dataloader('train', config)
    tr_agent.dataset = val_dataset

    # start training
    clock = tr_agent.clock
    min_cd_t = 999
    min_epoch = 0
    
    

    for e in range(clock.epoch, config.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            tr_agent.train_func(data)

            # visualize
            if config.vis and clock.step % config.vis_frequency == 0:
                tr_agent.visualize_batch(data, "train")

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            losses = tr_agent.collect_loss()
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            clock.tick()

        #-------------------------------
        cd_t_coarse = tr_agent.eval_one_epoch()
        if cd_t_coarse < min_cd_t:
            min_cd_t = cd_t_coarse
            min_epoch = e
            tr_agent.save_ckpt('best')
            print(min_epoch, min_cd_t)
        with open("min_cd/"+config.category+".txt", "a") as f:
            string = str(e) + ", now_cd_t: coarse, " + str(10000*cd_t_coarse) + \
                "; min_epoch, " + str(min_epoch) + "; min_cd_t: " + str(10000*min_cd_t) + "\n"
            f.write(string)
        #--------------------------------

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % config.save_frequency == 0:
            tr_agent.save_ckpt()
        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
