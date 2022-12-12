import torch
import numpy as np
import os
'''
path_a = "./data/Mannequin_data_Public/12_15_2020_16:40:33_blue_knit_old/"
path_b = "./data/Mannequin_data_Public/12_24_2020_20:24:48_dia_fixed_body_size/"
path_c = "./data/Mannequin_data_Public/12_10_2020_15:12:05_measurement_fixed_bar_recentered/"
'''
from model.heatmap2motor import Heatmap2Motor
from datasets.garment2motor import Garment2Motor



class MotorBasedRetriever:
    def __int__(self):
        self.heatmap2motor=self.model_init()
        self.g2m=self.dataset_init()

    def retrieve_by_motor(self,motor_p):
        idx=self.g2m.retrieve_by_motor(motor_p)
        image=self.g2m.get_image(idx)
        return image

    def heatmap_2_motor(self,heatmap):
        motor_p=self.heatmap2motor(heatmap)
        return motor_p

    def dataset_init(self):
        path_a = "./data/Mannequin_data_Public/12_15_2020_16:40:33_blue_knit_old/"
        g2m = Garment2Motor(path_a,split=False)
        return g2m

    def model_init(self):
        name = 'garmentABC'
        ckpt_path = os.path.join('./ckpt', name)
        # os.makedirs(ckpt_path, exist_ok=True)
        model = Heatmap2Motor()
        assert (os.path.exists(os.path.join(ckpt_path, 'best_model.pth'))), "Ckpt not found!"
        print("Found ckpt, loading...")
        model.load_state_dict(torch.load(os.path.join(ckpt_path, 'latest_model.pth')))
        if torch.cuda.is_available():
            model=model.cuda()
        return model

