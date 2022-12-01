import torch
from networks import get_network, set_requires_grad
from agent.base import GANzEAgent
from util.emd import earth_mover_distance
import sys
import os
import torch.nn as nn

sys.path.append('net/util/ChamferDistancePytorch')
from chamfer3D import dist_chamfer_3D
chamLoss = dist_chamfer_3D.chamfer_3DDist()

class Discriminators(nn.Module):
    def __init__(self, config=None, num=4):
        super(Discriminators, self).__init__()
        self.D_list = [get_network(config, "S_D").cuda() for i in range(num)]
        self.D_list = torch.nn.Sequential(*self.D_list)
    
    def forward(self, i, latent):
        output = self.D_list[i](latent)
        return output

class KTAgent(GANzEAgent):
    def __init__(self, config):
        super(KTAgent, self).__init__(config)

    def build_net(self, config):
        self.encoder = get_network(config, "KT_encoder").cuda()
        self.decoder = get_network(config, "KT_decoder").cuda()
        self.config = config

        self.is_train = config.is_train
        self.is_refine = False

        if self.is_train == False:
            self.cd_p_show = 0
            self.cd_t_show = 0
            self.emd_show = 0
            self.item_num = 0

        else:
            self.net_D = Discriminators(config, 2).cuda()

    def collect_loss(self):
        loss_dict = {"loss_D": self.loss_D,
                     "loss_G": self.loss_G_fake,
                    "gt_coarse(emd)": self.source_coarse_loss,
                    "gt(emd)": self.source_loss,
                    "p_coarse(ucd_p)": self.target_coarse_loss,
                    "partial(ucd_p)": self.target_loss
                    }
        return loss_dict

    def writePoints(self, filename, points_input):
        with open(filename, "w") as f:
            for index in range(points_input.shape[0]):
                x,y,z = points_input[index]
                f.write(str(x)+" "+str(y)+" "+str(z)+ "\n")

    def get_latent(self):
        with torch.no_grad():
            self.source_latent = self.encoder(self.real_pc, "source")
            _, _, self.real_latent = self.decoder(self.source_latent, "source")
            self.real_latent.append(self.source_latent)
            self.fake_latent.append(self.target_latent)
            return self.fake_latent, self.real_latent

    def forward(self, data):
        self.raw_pc = data['raw'].cuda()
        self.real_pc = data['real'].cuda()
        
        if self.is_train == True:
            self.source_latent = self.encoder(self.real_pc)
            self.target_latent = self.encoder(self.raw_pc)

            self.source_coarse, self.source_output, self.real_latent = self.decoder(self.source_latent, "source")
            self.target_coarse, self.target_output, self.fake_latent = self.decoder(self.target_latent, "target")
        self.forward_GE()

    def forward_GE(self):
        if self.is_train == False:
            self.target_latent = self.encoder(self.raw_pc)
            _, self.target_output, _ = self.decoder(self.target_latent, "target")
            
            self.cd_p_show += self.cd_p_loss(self.real_pc, self.target_output)
            self.cd_t_show += self.cd_t_loss(self.real_pc, self.target_output)

            self.item_num += 1
            if self.item_num % 50 == 0:
                print("target", self.item_num, self.show_loss())

    def show_loss(self):
        return 0,\
                    self.cd_t_show.detach().cpu().numpy()/self.item_num,\
                    self.cd_p_show.detach().cpu().numpy()/self.item_num,

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.base_lr = config.lr
        if self.is_train == True:
            set_requires_grad(self.encoder, False)
            set_requires_grad(self.decoder, False)
            for name, para in self.decoder.named_parameters():
                if "correct_layer" in name:
                    para.requires_grad = True
            
            self.optimizer_decoder_Partial = torch.optim.Adam(filter(lambda p: p.requires_grad, self.decoder.parameters()), lr=self.config.lr, betas=(self.config.beta1_gan, 0.999))

            set_requires_grad(self.encoder, True)
            set_requires_grad(self.decoder, True)
            for name, para in self.decoder.named_parameters():
                if "correct_layer" in name:
                    para.requires_grad = False
            self.optimizer_decoder_GT = torch.optim.Adam(filter(lambda p: p.requires_grad, self.decoder.parameters()), lr=self.config.lr, betas=(self.config.beta1_gan, 0.999))

            set_requires_grad(self.encoder, True)
            set_requires_grad(self.decoder, True)

            self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=self.config.lr, betas=(self.config.beta1_gan, 0.999))

            #---------------------------------------------------------------------------
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=2*self.config.lr, betas=(self.config.beta1_gan, 0.999))
            #---------------------------------------------------------------------------
            self.set_scheduler(config)

    def set_scheduler(self, config):
        """set lr scheduler used in training"""
        self.scheduler_encoder = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_encoder, config.lr_decay)

        self.scheduler_decoder_GT = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_decoder_GT, config.lr_decay)
        self.scheduler_decoder_Partial = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_decoder_Partial, config.lr_decay)

        self.scheduler_D = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_D, config.lr_decay)

    def update_learning_rate(self):
        """record and update learning rate"""
        if not self.optimizer_encoder.param_groups[-1]['lr'] < self.base_lr / 5.0:
            self.scheduler_encoder.step(self.clock.epoch)

        if not self.optimizer_decoder_GT.param_groups[-1]['lr'] < self.base_lr / 5.0:
            self.scheduler_decoder_GT.step(self.clock.epoch)
        if not self.optimizer_decoder_Partial.param_groups[-1]['lr'] < self.base_lr / 5.0:
            self.scheduler_decoder_Partial.step(self.clock.epoch)

        if not self.optimizer_D.param_groups[-1]['lr'] < self.base_lr / 5.0:
            self.scheduler_D.step(self.clock.epoch)


    def backward_D(self):
        loss_D_fake = 0
        for idx in range(len(self.real_latent)):
            pred_fake = self.net_D(idx, self.fake_latent[idx].detach())
            fake = torch.zeros_like(pred_fake).fill_(0.0).cuda()
            loss_D_fake = self.criterionGAN(pred_fake, fake) + loss_D_fake
        self.loss_D_fake = loss_D_fake/(len(self.real_latent))

        loss_D_real = 0
        for idx in range(len(self.real_latent)):
            pred_real = self.net_D(idx, self.real_latent[idx].detach())
            real = torch.zeros_like(pred_real).fill_(1.0).cuda()
            loss_D_real = self.criterionGAN(pred_real, real) + loss_D_real
        self.loss_D_real = loss_D_real/(len(self.real_latent))

        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        self.loss_D.backward()

    def update_D(self):
        self.backward_D()

    def backward_EG_GT(self):  
        self.source_coarse_loss = earth_mover_distance(self.source_coarse, self.real_pc)
        self.source_loss = earth_mover_distance(self.source_output, self.real_pc)
        self.GT_loss = self.source_coarse_loss + self.source_loss

        self.GT_loss.backward(retain_graph=True)

    def backward_EG_Partial(self):
        loss_G_fake = 0
        for idx in range(len(self.real_latent)):
            pred_fake = self.net_D(idx, self.fake_latent[idx])
            real = torch.zeros_like(pred_fake).fill_(1.0).cuda()
            loss_G_fake = self.criterionGAN(pred_fake, real) + loss_G_fake
        self.loss_G_fake = loss_G_fake/(len(self.real_latent))
        
        self.target_loss = self.cd_p_loss(self.raw_pc, self.target_output, True)
        self.target_coarse_loss = self.cd_p_loss(self.raw_pc, self.target_coarse, True)

        a, b, c = [0.1, 1, 1]
        self.loss_EG = a * self.loss_G_fake + b * self.target_loss + c * self.target_coarse_loss
        
        self.loss_EG.backward(retain_graph=True)

    def update_G_and_E(self):
        self.backward_EG_GT()
        self.backward_EG_Partial()


    def optimize_network(self):
        self.optimizer_D.zero_grad()
        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder_GT.zero_grad()
        self.optimizer_decoder_Partial.zero_grad()
        self.update_G_and_E()
        self.update_D()
        self.optimizer_D.step()
        self.optimizer_encoder.step()
        self.optimizer_decoder_GT.step()
        self.optimizer_decoder_Partial.step()

    def get_point_cloud(self):
        """get real/fake/raw point cloud of current batch"""
        real_pts = self.real_pc.transpose(1, 2).detach().cpu().numpy()
        fake_pts = self.target_output.transpose(1, 2).detach().cpu().numpy()
        raw_pts = self.raw_pc.transpose(1, 2).detach().cpu().numpy()
        return raw_pts, fake_pts, real_pts

    def visualize_batch(self, data, mode, **kwargs):
        tb = self.train_tb if mode == 'train' else self.val_tb

        num = 2

        real_pts = data['real'][:num].transpose(1, 2).detach().cpu().numpy()
        fake_pts = self.target_output[:num].transpose(1, 2).detach().cpu().numpy()
        raw_pts = self.raw_pc[:num].transpose(1, 2).detach().cpu().numpy()
        source_output = self.source_output[:num].transpose(1, 2).detach().cpu().numpy()

        tb.add_mesh("real", vertices=real_pts, global_step=self.clock.step)
        tb.add_mesh("final", vertices=fake_pts, global_step=self.clock.step)
        tb.add_mesh("input", vertices=raw_pts, global_step=self.clock.step)
        tb.add_mesh("source_comp", vertices=source_output, global_step=self.clock.step)

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'encoder_state_dict': self.encoder.cpu().state_dict(),
            'decoder_state_dict':self.decoder.cpu().state_dict(),
            'netD_state_dict':self.net_D.cpu().state_dict(),
            'optimizer_decoder_GT_state_dict': self.optimizer_decoder_GT.state_dict(),
            'optimizer_encoder_state_dict': self.optimizer_encoder.state_dict(),
            'optimizer_decoder_Partial_state_dict': self.optimizer_decoder_Partial.state_dict(),
            'optimizer_netD_state_dict': self.optimizer_D.state_dict()
        }, save_path)

        self.encoder.cuda()
        self.decoder.cuda()
        self.net_D.cuda()

    def load_ckpt(self, config):
        """load checkpoint from saved checkpoint"""
        load_path = config.pretrain_path
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])
        if self.is_train == True:
            self.net_D.load_state_dict(checkpoint['netD_state_dict'])
            self.optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder_state_dict'])
            self.optimizer_decoder_GT.load_state_dict(checkpoint['optimizer_decoder_GT_state_dict'])
            self.optimizer_decoder_Partial.load_state_dict(checkpoint['optimizer_decoder_Partial_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_netD_state_dict'])

    def eval(self):
        """set G, D, E to eval mode"""
        self.encoder.eval()
        self.decoder.eval()

    def cd_p_loss(self, input1, input2, is_u = False):
        """
            input: B 3 N
            output: B
        """
        dist1, dist2, _, _ = chamLoss(input1.transpose(2, 1), input2.transpose(2, 1))
        if is_u == False:
            cd_p_show = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
        else:
            cd_p_show = torch.sqrt(dist1).mean(1)
        return torch.mean(cd_p_show)
    
    def cd_t_loss(self, input1, input2, is_u = False):
        """
            input: B 3 N
            output: B
        """
        dist1, dist2, _, _ = chamLoss(input1.transpose(2, 1), input2.transpose(2, 1))
        if is_u == False:
            cd_t_show = dist1.mean(1) + dist2.mean(1)
        else:
            cd_t_show = dist1.mean(1)
        return torch.mean(cd_t_show)

    def eval_one_epoch(self):
        self.encoder.eval()
        self.decoder.eval()

        cd_t = 0
        with torch.no_grad():
            for i in range(len(self.dataset)):
                raw_pc = self.dataset[i]['raw'].cuda().unsqueeze(0)
                real_pc = self.dataset[i]['real'].cuda().unsqueeze(0)

                target_latent = self.encoder(raw_pc)
                _, target_output, _ = self.decoder(target_latent, "target")

                dist1, dist2, _, _ = chamLoss(real_pc.transpose(2, 1), target_output.transpose(2, 1))
                cd_t += (dist1.mean(1) + dist2.mean(1))

            cd_t = (cd_t/len(self.dataset))[0].cpu().numpy()

        self.encoder.train()
        self.decoder.train()
        
        return cd_t
