import warnings
import numpy as np
from dcem_model import Model
#from calculate_flow import FlowNet2Utils
import torch
import os
from photo_error import mse_
from calculate_flow_neuflow import NeuFlowUtils
import cv2



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
np.random.seed(0)
warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

'''
 def __init__(self,
                 img_goal: np.ndarray,
                 ct=20,
                 horizon=10,
                 LR=0.005,
                 iterations=1,
                 ):
'''
class Rtvs:
    '''
    Code for RTVS: Real-Time Visual Servoing, IROS 2021
    '''
    def __init__(self,
                 img_goal: np.ndarray,
                 ct=20,
                 horizon=10,
                 LR=0.005,
                 iterations=1,
                 ):
        '''
        img_goal: RGB array for final pose
        ct = image downsampling parameter (high ct => faster but less accurate)
        LR = learning rate of NN
        iterations = iterations to train NN (high value => slower but more accurate)
        horizon = MPC horizon
        '''
        self.img_goal = img_goal
        self.horizon = horizon
        self.iterations = iterations
        self.ct = ct
        #self.flow_utils = FlowNet2Utils()
        self.flow_utils = NeuFlowUtils()
        self.vs_lstm = Model().to(device="cuda:0")
        self.optimiser = torch.optim.Adam(self.vs_lstm.parameters(),
                                          lr=LR, betas=(0.93, 0.999))
        self.loss_fn = torch.nn.MSELoss(size_average=False)

    
    def get_vel(self, img_src, pre_img_src):
        '''
            img_src = current RGB camera image
            prev_img_src = previous RGB camera image 
                        (to be used for depth estimation using flowdepth)
        '''
        img_goal = self.img_goal
        flow_utils = self.flow_utils
        vs_lstm = self.vs_lstm
        loss_fn = self.loss_fn
        optimiser = self.optimiser
        ct = self.ct

        photo_error_val = mse_(img_src, img_goal)
        if photo_error_val < 6000 and photo_error_val > 3600:
            self.horizon = 10*(photo_error_val/6000)
        elif photo_error_val < 3000:
            self.horizon = 6

        target_width = 512
        target_height = 384
        f12 = flow_utils.flow_calculate(img_src, img_goal)[::ct, ::ct]
        flow_depth_proxy = flow_utils.flow_calculate(
            img_src, pre_img_src).astype('float64')

        Cy, Cx = flow_depth_proxy.shape[1]/2, flow_depth_proxy.shape[0]/2
        flow_depth = np.linalg.norm(flow_depth_proxy[::ct, ::ct], axis=2)
        flow_depth = flow_depth.astype('float64')
        vel, Lsx, Lsy = get_interaction_data(
            0.6*(1/(1+np.exp(-1/flow_depth)) - 0.5), ct, Cy, Cx)

        Lsx = torch.tensor(Lsx, dtype=torch.float32).to(device="cuda:0")
        Lsy = torch.tensor(Lsy, dtype=torch.float32).to(device="cuda:0")
        f12 = torch.tensor(f12, dtype=torch.float32).to(device="cuda:0")
        f12 = vs_lstm.pooling(f12.permute(2, 0, 1).unsqueeze(dim=0))

        for itr in range(self.iterations):
            vs_lstm.v_interm = []
            vs_lstm.f_interm = []
            vs_lstm.mean_interm = []

            vs_lstm.zero_grad()
            f_hat = vs_lstm.forward(vel, Lsx, Lsy, self.horizon, f12)
            loss = loss_fn(f_hat, f12)

            print("MSE:", str(np.sqrt(loss.item())))
            loss.backward(retain_graph=True)
            optimiser.step()

        #Do not accumulate flow and velocity at train time
        vs_lstm.v_interm = []
        vs_lstm.f_interm = []
        vs_lstm.mean_interm = []

        f_hat = vs_lstm.forward(vel, Lsx, Lsy, -self.horizon,
                                f12.to(torch.device('cuda:0')))
        
        vel = vs_lstm.v_interm[0].detach().cpu().numpy()

        return vel


def get_interaction_data(d1, ct, Cy, Cx):
    ky = Cy
    kx = Cx
    xyz = np.zeros([d1.shape[0], d1.shape[1], 3])
    Lsx = np.zeros([d1.shape[0], d1.shape[1], 6])
    Lsy = np.zeros([d1.shape[0], d1.shape[1], 6])

    med = np.median(d1)
    xyz = np.fromfunction(lambda i, j, k: 0.5*(k-1)*(k-2)*(ct*j-float(Cx))/float(kx) - k*(k-2)*(ct*i-float(Cy))/float(ky) + 0.5*k*(
        k-1)*((d1[i.astype(int), j.astype(int)] == 0)*med + d1[i.astype(int), j.astype(int)]), (d1.shape[0], d1.shape[1], 3), dtype=float)

    Lsx = np.fromfunction(lambda i, j, k: (k == 0).astype(int) * -1/xyz[i.astype(int), j.astype(int), 2] + (k == 2).astype(int) * xyz[i.astype(int), j.astype(int), 0]/xyz[i.astype(int), j.astype(int), 2] + (k == 3).astype(int) * xyz[i.astype(
        int), j.astype(int), 0]*xyz[i.astype(int), j.astype(int), 1] + (k == 4).astype(int)*(-(1+xyz[i.astype(int), j.astype(int), 0]**2)) + (k == 5).astype(int)*xyz[i.astype(int), j.astype(int), 1], (d1.shape[0], d1.shape[1], 6), dtype=float)

    Lsy = np.fromfunction(lambda i, j, k: (k == 1).astype(int) * -1/xyz[i.astype(int), j.astype(int), 2] + (k == 2).astype(int) * xyz[i.astype(int), j.astype(int), 1]/xyz[i.astype(int), j.astype(int), 2] + (k == 3).astype(int) * (1+xyz[i.astype(
        int), j.astype(int), 1]**2) + (k == 4).astype(int)*-xyz[i.astype(int), j.astype(int), 0]*xyz[i.astype(int), j.astype(int), 1] + (k == 5).astype(int) * -xyz[i.astype(int), j.astype(int), 0], (d1.shape[0], d1.shape[1], 6), dtype=float)

    return None, Lsx, Lsy

