import sys
import tools.init_paths
from core.inference import get_max_preds
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch


def ImageCaption(image: np.array, caption: str, color=(255, 255, 255), font_size=40) -> np.array:
    h = image.shape[0]
    w = image.shape[1]
    img = Image.fromarray(image)
    font = ImageFont.truetype("./fonts/Arial.ttf", font_size)
    draw = ImageDraw.Draw(img)

    draw.text((w // 2, h - h // 16), caption, color, font=font, anchor="ms")
    img = np.array(img)
    return img


def GetLandmarkCoord(heatmaps: torch.Tensor):
    preds, maxvals = get_max_preds(heatmaps.cpu().detach().numpy())
    xs = []
    ys = []
    colors = []
    top = np.array((0, 0, 1.0))
    bottom = np.array((1.0, 0, 0))
    for i in range(len(maxvals[0])):
        xs.append(preds[0][i][0])
        ys.append(preds[0][i][1])
        colors.append(maxvals[0][i] * top + (1 - maxvals[0][i]) * bottom)
        colors[i] = np.multiply(colors[i], 255).astype(np.uint8)
    return xs, ys, colors


def LandmarkAnnotation(image: np.array, heatmaps: torch.Tensor, color=(255, 255, 255)) -> np.array:
    xs, ys, colors = GetLandmarkCoord(heatmaps)
    # img=np.multiply(image, 255).astype(np.uint8)
    #print(image.shape)
    img = Image.fromarray(image)
    font = ImageFont.truetype("./fonts/Arial.ttf", 20)
    draw = ImageDraw.Draw(img)
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        r = 5
        draw.text((x + r, y - 5 * r), str(i + 1), color, font=font)
        draw.ellipse(((x - r, y - r), (x + r, y + r)), fill=tuple(colors[i].tolist()))
    img = np.array(img)
    return img
