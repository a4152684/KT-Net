import torch
import torch.nn as nn
import math
import torch.nn.functional as F

#------------------------------------------------------------------------------
class KRA(nn.Module):
    def __init__(self, channel):
        super(KRA, self).__init__()
        self.fc1 = nn.Linear(channel, channel, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel, channel, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def gen_grid_up(up_ratio, grid_size=0.2):
    sqrted = int(math.sqrt(up_ratio)) + 1
    for i in range(1, sqrted + 1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio // i
            break

    grid_x = torch.linspace(-grid_size, grid_size, steps=num_x)
    grid_y = torch.linspace(-grid_size, grid_size, steps=num_y)

    x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
    grid = torch.stack([x, y], dim=-1).view(-1, 2).transpose(0, 1).contiguous()
    return grid

class KT_encoder(nn.Module):
    """
        缺少SA模块
    """
    def __init__(self, output_size=1024):
        super(KT_encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = self.conv1(x)
        x = F.relu(x) # 128

        x = self.conv2(x) # 256
        global_feature, _ = torch.max(x, 2) #256
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1) #512

        x = self.conv3(x)
        x = F.relu(x) # 512

        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        return global_feature.view(batch_size, -1)
    
class KT_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, scale, cat_feature_num):
        super(KT_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.scale = scale
        self.grid = gen_grid_up(2 ** (int(math.log2(scale))), 0.05).cuda().contiguous()
        self.conv1 = nn.Conv1d(cat_feature_num, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

        self.correct_layer1 = KRA(1024)
        self.correct_layer2 = KRA(1024)

    def forward(self, x, phase = "source"):
        latent = []

        batch_size = x.size()[0]
        if phase == "source":
            coarse = F.relu(self.fc1(x))
            latent.append(coarse)
            coarse = F.relu(self.fc2(coarse))
            latent.append(coarse)
            coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)
        elif phase == "target":
            coarse = self.fc1(x)
            coarse = self.correct_layer1(coarse) + coarse
            coarse = F.relu(coarse)
            latent.append(coarse)

            coarse = self.fc2(coarse)
            coarse = self.correct_layer2(coarse) + coarse
            coarse = F.relu(coarse)
            latent.append(coarse)

            coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)

        grid = self.grid.clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous().cuda()

        point_feat = (
            (coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                3)).transpose(1,
                                                                                                              2).contiguous()

        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)

        feat = torch.cat((grid_feat, point_feat, global_feat), 1)

        center = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                      3)).transpose(1,
                                                                                                                    2).contiguous()

        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        return coarse, fine, latent
