from transformers import AutoModel
import torch
import timm


class XcitDinoEncoder(torch.nn.Module):

    def __init__(self, 
            timm_model='xcit_small_12_p8_224',
            fb_model='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_cp8_dino.pth',
            device='cuda'
        ):
        super().__init__()
        net = timm.create_model(timm_model, num_classes=0, pretrained=False)
        net.to(device)
        checkpoint = torch.hub.load_state_dict_from_url(fb_model, map_location=device, check_hash=True)
        checkpoint = timm.models.xcit.checkpoint_filter_fn(checkpoint, net)
        net.load_state_dict(checkpoint, strict=True)
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x

    @classmethod
    def load(cls, checkpoint):
        ptnet = cls()
        ptnet.load_state_dict(torch.load(checkpoint))
        return ptnet


class ProjectionHead(torch.nn.Module):

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        return self.model(x)


def AutoEncoderFactory(backend, modelpath):

    if backend == "timm":

        class AutoEncoder(torch.nn.Module):

            def __init__(self, model=modelpath, device='cuda'):
                super().__init__()
                net = timm.create_model(model, num_classes=0, pretrained=True)
                net.to(device)
                self.net = net

            def forward(self, x):
                x = self.net(x)
                return x 

            @classmethod
            def load(cls, checkpoint):
                ptnet = cls()
                ptnet.load_state_dict(torch.load(checkpoint))
                return ptnet

    elif backend == "hf":

        class AutoEncoder(torch.nn.Module):

            def __init__(self, model=modelpath, device='cuda'):
                super().__init__()
                net = AutoModel.from_pretrained(model)
                net.to(device)
                self.net = net

            def forward(self, x):
                x = self.net(x)
                x = x.last_hidden_state[:,0,:]
                return x

            @classmethod
            def load(cls, checkpoint):
                ptnet = cls()
                ptnet.load_state_dict(torch.load(checkpoint))
                return ptnet

    else:
        
        raise NotImplementedError

    return AutoEncoder
