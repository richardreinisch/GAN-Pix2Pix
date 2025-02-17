
import os
import numpy as np

import torch
import torch.nn as nn
import functools

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IMG_WIDTH = 256
IMG_HEIGHT = 256


def normalize(inp, tar):
    input_image = (inp / 127.5) - 1
    target_image = (tar / 127.5) - 1
    return input_image, target_image

def read_image(image):

    image = np.array(image)
    width = image.shape[1]
    width_half = width // 2

    input_image = image[:, :width_half, :]
    target_image = image[:, width_half:, :]

    input_image = input_image.astype(np.float32)
    target_image = target_image.astype(np.float32)

    return input_image, target_image

def read_single_image(image):

    image = image.resize((256, 256))
    image = np.array(image)
    image = image.astype(np.float32)

    return image

def get_norm_layer():
    norm_type = 'batch'
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    return norm_layer


class Val_Normalize(object):
    def __call__(self, image):
        inp, tar = read_image(image)
        inp, tar = normalize(inp, tar)
        image_a = torch.from_numpy(inp.copy().transpose((2,0,1)))
        image_b = torch.from_numpy(tar.copy().transpose((2,0,1)))
        return image_a, image_b

class Just_Read(object):
    def __call__(self, image):
        input = read_single_image(image)
        return torch.from_numpy(input.copy().transpose((2, 0, 1)))

class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        down_conv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = norm_layer(inner_nc)

        up_relu = nn.ReLU(True)
        up_norm = norm_layer(outer_nc)

        if outermost:
            up_conv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            up_conv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
            model = down + up
        else:
            up_conv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [down_relu, down_conv, down_norm]
            up = [up_relu, up_conv, up_norm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, nf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):

        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer

        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(nf * 4, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf * 2, nf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf, nf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        return self.model(input)


device = torch.device('cuda')
n_gpus = 1
batch_size = 32 * n_gpus

norm_layer = get_norm_layer()

generator = UnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False)#.cuda().float()
# generator.apply(weights_init)

device = 'cuda'
# device = 'cpu'

VAL_DIR = 'test/val/'

val_ds = ImageFolder(VAL_DIR, transform=transforms.Compose([Val_Normalize()]))
val_dl = DataLoader(val_ds, batch_size)

generator = torch.nn.DataParallel(generator)  # multi-GPUs

generator.load_state_dict(torch.load('model/generator_epoch_9.pth'))
generator.eval()
generator = generator.to(device)

for (inputs, targets), _ in val_dl:
    inputs = inputs.to(device)
    generated_output = generator(inputs)
    save_image(generated_output.data[:10], 'test/result/sample_generated.jpg', nrow=5, normalize=True)

print("Finished :-)")
  