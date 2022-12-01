from networks.networks_kt import KT_encoder, KT_decoder
from networks.networks_gan import SDiscriminatorFC
def get_network(config, name):
    if name == "KT_encoder":
        return KT_encoder(config.latent_dim)
    elif name == "KT_decoder":
        #return KT_decoder(2048,16384,8,1024+2+3)
        return KT_decoder(2048,2048,1,1024+2+3)
    elif name == "S_D":
        return SDiscriminatorFC(config.latent_dim, config.D_features)
    else:
        raise NotImplementedError("Got name '{}'".format(name))


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
