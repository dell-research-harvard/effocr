from transformers import AutoModelForImageClassification
import torch
import timm
from torch.nn import CrossEntropyLoss


class XcitDinoClassifier(torch.nn.Module):

    def __init__(self,  
            n_classes,           
            timm_model='xcit_small_12_p8_224',
            fb_model='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_cp8_dino.pth',
            device='cuda'
        ):
        super().__init__()
        net = timm.create_model(timm_model, num_classes=n_classes, pretrained=False)
        net.to(device)
        checkpoint = torch.hub.load_state_dict_from_url(fb_model, map_location=device, check_hash=True)
        checkpoint = timm.models.xcit.checkpoint_filter_fn(checkpoint, net)
        net.load_state_dict(checkpoint, strict=False)
        self.net = net
        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        x = self.net(x)
        return x 

    @classmethod
    def load(cls, checkpoint, n_classes):
        ptnet = cls(n_classes=n_classes)
        ptnet.load_state_dict(torch.load(checkpoint))
        return ptnet


def AutoClassifierFactory(backend, modelpath, n_classes):

    if backend == "timm":

        class AutoClassifier(torch.nn.Module):

            def __init__(self, model=modelpath, device='cuda'):
                super().__init__()
                net = timm.create_model(model, num_classes=n_classes, pretrained=True)
                net.to(device)
                self.net = net
                self.criterion = CrossEntropyLoss()

            def forward(self, x):
                x = self.net(x)
                return x 

            @classmethod
            def load(cls, checkpoint):
                ptnet = cls()
                ptnet.load_state_dict(torch.load(checkpoint))
                return ptnet

    elif backend == "hf":

        class AutoClassifier(torch.nn.Module):

            def __init__(self, model=modelpath, device='cuda'):
                super().__init__()
                net = AutoModelForImageClassification.from_pretrained(model, num_labels=n_classes)
                net.to(device)
                self.net = net
                self.criterion = CrossEntropyLoss()

            def forward(self, x):
                x = self.net(x)
                return x

            @classmethod
            def load(cls, checkpoint):
                ptnet = cls()
                ptnet.load_state_dict(torch.load(checkpoint))
                return ptnet

    else:
        
        raise NotImplementedError

    return AutoClassifier
