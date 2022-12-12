import torchvision.transforms as T
import torch
import torchvision.transforms.functional as F

class RandomAffineBatch(T.RandomAffine):
    def __int__(self,*args):
        super(RandomAffineBatch,self).__init__(*args)

    def forward(self, imgs):
        channels, height, width = F.get_dimensions(imgs[0])
        img_size = [width, height]  # flip for keeping BC on get_params call
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        results=[]
        for img in imgs:
            fill = self.fill
            channels, height, width = F.get_dimensions(img)
            if isinstance(img, torch.Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * channels
                else:
                    fill = [float(f) for f in fill]
            results.append(F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center))
        return results