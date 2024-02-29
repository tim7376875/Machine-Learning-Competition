import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
#from torchvision import models
from resnetmodel import *
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils import GradCAM, show_cam_on_image, center_crop_img
from imgaug import augmenters as iaa


def main():
    model = ResNet18()
    target_layers = [model.layer4[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    seq = iaa.Sequential([
        # iaa.Resize((224, 224)),
        # iaa.LinearContrast((10)),  # crop images from each side by 0 to 16px (randomly chosen)
        # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        # iaa.Multiply((1.5, 1.5), per_channel=0.5),
        # iaa.Sharpen(alpha=(0.5, 0.5), lightness=(1, 1)),
        #iaa.AdditiveGaussianNoise(loc=1, scale=(20, 40), per_channel=0.5),
        #iaa.GaussianBlur(sigma=(0.5, 2)),
        # iaa.Multiply((1.2, 1.4)),
        iaa.LinearContrast((2.5, 2.5)),
        iaa.Sharpen(alpha=(0.5, 0.5), lightness=(1.1, 1.1)),
        # iaa.EdgeDetect(),
    ])

    data_transform = transforms.Compose([
                                         #transforms.Resize(224),
                                         transforms.RandomResizedCrop(size=((640, 480)), scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
                                         transforms.RandomHorizontalFlip(0.5),
                                         transforms.RandomRotation(degrees=30),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.481, 0.423, 0.368], std=[0.247, 0.241, 0.249]),
                                         ])
    # load image
    img_path = "/home/users/EPARC/EPARC2/val_data/02nehv1tf6.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)
    val_img = np.transpose(img, (2, 0, 1))
    val_images_aug = seq(images=val_img)
    val_images = np.transpose(val_images_aug, (1, 2, 0))
    img_PIL = transforms.ToPILImage()(val_images)

    # [C, H, W]
    img_tensor = data_transform(img_PIL)

    def showpic(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        plt.imshow(inp)
        plt.pause(0.001)
        plt.clf()


    showpic(img_tensor)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 105  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()