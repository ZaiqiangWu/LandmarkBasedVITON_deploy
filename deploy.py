import torch

import os
from options.test_options import TestOptions
from model.pix2pixHD.models import create_model
from util.garment_heatmap import HeatmapGenerator
import torchvision.transforms as transforms
import cv2
import numpy as np


def make_pix2pix_model(name, nc):
    opt = TestOptions().parse(save=False, use_default=True, show_info=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.name = name
    opt.input_nc = nc
    opt.isTrain = False
    model = create_model(opt)
    # print(model)
    return model


def make_landmark_detector():
    return HeatmapGenerator()


class VirtualGarmentSynthesizer:
    def __init__(self):
        self.download_pretrained_weight()
        self.landmark_detector = make_landmark_detector()
        self.model_k2qm = make_pix2pix_model('keypoint2qmeasurment', 25)
        self.model_qm2t = make_pix2pix_model('qmeasurement2target', 3)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        resize = transforms.Resize((384, 288))
        self.pre_transforms = transforms.Compose([normalize, resize])
        self.post_transform = transforms.Resize((512, 512))

    def forward(self, im: torch.Tensor):
        im = self.pre_transforms(im)
        with torch.no_grad():
            heatmaps = self.post_transform(self.landmark_detector(im))
            qmeasurement = self.model_k2qm.inference(heatmaps)
            target_garment = self.model_qm2t.inference(qmeasurement)
        return qmeasurement, target_garment

    def download_pretrained_weight(self):
        ckpt_k2qm = 'checkpoints/keypoint2qmeasurment/latest_net_G.pth'
        if not os.path.exists(ckpt_k2qm):
            print("Model not found, downloading from google drive...")
            import gdown
            path, name = os.path.split(ckpt_k2qm)
            os.makedirs(path, exist_ok=True)
            # https://drive.google.com/file/d/13vXDE5bbA0LdfChxwl9-_hiMS3fgAwjE/view?usp=sharing
            id = "13vXDE5bbA0LdfChxwl9-_hiMS3fgAwjE"
            output = ckpt_k2qm
            gdown.download(id=id, output=output, quiet=False)
        ckpt_qm2t = 'checkpoints/qmeasurement2target/latest_net_G.pth'
        if not os.path.exists(ckpt_qm2t):
            print("Model not found, downloading from google drive...")
            import gdown
            path, name = os.path.split(ckpt_qm2t)
            os.makedirs(path, exist_ok=True)
            # https://drive.google.com/file/d/13vIVbAvUEXUWP3F30dFStw_LerG9Wifo/view?usp=sharing
            id = "13vIVbAvUEXUWP3F30dFStw_LerG9Wifo"
            output = ckpt_qm2t
            gdown.download(id=id, output=output, quiet=False)


if __name__ == '__main__':
    im = cv2.imread('./example_input.png')
    im = torch.from_numpy(im) / 255.0
    im = im.permute(2, 0, 1)  # CHW, BGR
    if torch.cuda.is_available():
        im = im.cuda()
    im = im.unsqueeze(0)
    viton = VirtualGarmentSynthesizer()
    _, target_garment = viton.forward(im)
    target_garment = target_garment[0]
    target_garment = target_garment.permute(1, 2, 0)
    target_garment = target_garment.cpu().numpy()
    target_garment = np.clip(target_garment, 0, 1)
    target_garment = (target_garment * 255).astype(np.uint8)
    cv2.imwrite('./example_output.jpg', target_garment)
