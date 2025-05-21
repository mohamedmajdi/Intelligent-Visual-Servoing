import torch
import numpy
import cv2
from NeuFlow_v2.NeuFlow.neuflow import NeuFlow
from NeuFlow_v2.NeuFlow.backbone_v7 import ConvBlock 

class NeuFlowUtils:
    def __init__(self, image_height=432, image_width=768):
        self.image_height = image_height
        self.image_width = image_width
        self.device = torch.device('cuda')
        self.model = NeuFlow.from_pretrained("Study-is-happy/neuflow-v2").to(self.device)
        
        # Fuse Conv-BN layers
        for m in self.model.modules():
            if isinstance(m, ConvBlock):
                m.conv1 = NeuFlowUtils.fuse_conv_and_bn(m.conv1, m.norm1)
                m.conv2 = NeuFlowUtils.fuse_conv_and_bn(m.conv2, m.norm2)
                delattr(m, "norm1")
                delattr(m, "norm2")
                m.forward = m.forward_fuse
        
        self.model.eval()
        self.model.half()
        self.model.init_bhwd(1, self.image_height, self.image_width, 'cuda')

    def fuse_conv_and_bn(conv, bn):
        """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
        fusedconv = (
            torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv
    
    def preprocess(self, img):
        img_resized = cv2.resize(img, (self.image_width, self.image_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img_rgb=img_resized
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).half()
        return tensor[None].to(self.device)

    def flow_calculate(self, img1, img2):
        # img1, img2: np.ndarray (H,W,3), uint8
        t1 = self.preprocess(img1)
        t2 = self.preprocess(img2)
        with torch.no_grad():
            flow = self.model(t1, t2)[-1][0]  # shape: (2, H, W)
            flow = flow.permute(1,2,0).float().cpu().numpy()  # (H,W,2)
        return flow
