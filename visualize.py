import sys
import os
import requests
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import models_mae
import models_vit
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utility import remove_black_borders_fast
import utility

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=32)
    plt.axis('off')
    return

def save_image(image, save_path):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    image = torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).detach().numpy().astype(np.uint8)
    Image.fromarray(image).save(save_path)

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def prepare_ft_model(chkpt_dir, arch='vit_large_patch16', model_type='DR'):
    if 'DR' in model_type:        
        nb_classes = 5
    elif 'Glaucoma_PAPILA' in model_type:
        nb_classes = 3
    elif 'Glaucoma_Glaucoma_Fundus' in model_type:
        nb_classes = 3
    elif 'Glaucoma_ORIGA' in model_type:
        nb_classes = 2    
    elif 'AMD_AREDS' in model_type:
        nb_classes = 4
    elif 'Multi_Retina' in model_type:
        nb_classes = 4
    elif 'Multi_JSIEC' in model_type:
        nb_classes = 39

    model = models_vit.__dict__[arch](
        num_classes=nb_classes,
        drop_path_rate=0.1,
        global_pool=True,
    )
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    print("Resume checkpoint %s" % chkpt_dir)

    model.eval()
    return model

def run_one_image(img, model):

    x = torch.tensor(img)
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.25)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]
    
    # img = torch.cat([x[0], im_masked[0], y[0], im_paste[0]])

    plt.subplot(2, 2, 1)
    show_image(x[0], "original")
    # save_image(x[0], save_path='original.png')

    plt.subplot(2, 2, 2)
    show_image(im_masked[0], "masked")
    # save_image(im_masked[0], save_path='masked.png')

    plt.subplot(2, 2, 3)
    show_image(y[0], "reconstruction")
    # save_image(y[0], save_path='reconstruction.png')

    plt.subplot(2, 2, 4)
    show_image(im_paste[0], "reconstruction + visible")
    # save_image(im_paste[0], save_path='reconstruction_visible.png')

    # plt.show()
    plt.savefig('mae.png')

def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result
    
def run_cam(img, cam, label=None):    
    input_img = img - imagenet_mean
    input_img = img / imagenet_std    
    x = torch.tensor(input_img)
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x).float()
    target_category = None # 可以指定一个类别，或者使用 None 表示最高概率的类别
    input_tensor = x
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_category, aug_smooth=True)
    grayscale_cam = grayscale_cam[0, :]

    # 将 grad-cam 的输出叠加到原始图像上
    visualization = show_cam_on_image(img, grayscale_cam, image_weight=0.5)
    Image.fromarray(visualization).save('cam.png')

def run_classification(img, model, type):
    x = utility.prepare_data(img)
    # model inference    
    with torch.no_grad():
        output = model(x)
    output = nn.Softmax(dim=1)(output)
    output = output.squeeze(0).cpu().detach().numpy()

    if 'DR' in type:        
        # visualization
        categories = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        prob_result = utility.draw_result(output, categories, colors)
    
    elif 'Glaucoma_PAPILA' in type:
        # visualization
        categories = ['No glaucoma', 'Suspected glaucoma', 'Glaucoma']
        colors = ['blue', 'green', 'red']
        prob_result = utility.draw_result(output, categories, colors)    

    elif 'Glaucoma_Glaucoma_Fundus' in type:
        # visualization
        categories = ['No glaucoma', 'Early glaucoma', 'Advanced glaucoma']
        colors = ['blue', 'green', 'red']
        prob_result = utility.draw_result(output, categories, colors)

    elif 'Glaucoma_ORIGA' in type:
        # visualization
        categories = ['No glaucoma', 'Glaucoma']
        colors = ['blue', 'red']
        prob_result = utility.draw_result(output, categories, colors)
    
    elif 'AMD_AREDS' in type:
        # visualization
        categories = ['Non AMD', 'Mild AMD', 'Moderate AMD', 'Advanced AMD']
        colors =  ['blue', 'green', 'red', 'orange']
        prob_result = utility.draw_result(output, categories, colors)

    elif 'Multi_Retina' in type:
        # visualization
        categories = ['Normal', 'Cataract', 'Glaucoma', 'Others']
        colors =  ['blue', 'green', 'red', 'orange']
        prob_result = utility.draw_result(output, categories, colors)

    elif 'Multi_JSIEC' in type:
        # visualization
        categories = ['Normal', 'Tessellated fundus', 'Large optic cup', 'DR1',  'DR2', 'DR3', \
                      'BRVO', 'CRVO', 'RAO', 'Rhegmatogenous RD', 'CSCR', 'VKH disease', 'Maculopathy', \
                        'ERM', 'MH', 'Pathological myopia', 'Possible glaucoma', 'Optic atrophy', \
                        'Severe hypertensive retinopathy', 'Disc swelling and elevation', 'Dragged Disc', \
                        'Congenital disc abnormality', 'Retinitis pigmentosa', 'Bietti crystalline dystrophy', \
                        'Peripheral retinal degeneration and break', 'Myelinated nerve fiber', 'Vitreous particles', \
                        'Fundus neoplasm', 'Massive hard exudates', 'Yellow-white spots-flecks', 'Cotton-wool spots', \
                        'Vessel tortuosity', 'Chorioretinal atrophy-coloboma', 'Preretinal hemorrhage', 'Fibrosis', \
                        'Laser Spots', 'Silicon oil in eye', 'Blur fundus without PDR', 'Blur fundus with suspected PDR']
        
        colors =  ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', \
                   'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', \
                    'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', \
                    'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', \
                    'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', \
                    'darkslategrey', 'darkturquoise']

        prob_result = utility.draw_result(output, categories, colors)

    Image.fromarray(prob_result).save('classification.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mae', choices=['mae', 'cam', 'classification'])
    parser.add_argument('--img_path', type=str, default='./exampledata/DR/APTOS2019.png')
    parser.add_argument('--ft_model', type=str, default='DR_APTOS2019', choices=['DR_APTOS2019','DR_IDRID', \
                                                                                     'DR_MESSIDOR2','Glaucoma_PAPILA', \
                                                                                     'Glaucoma_Glaucoma_Fundus','Glaucoma_ORIGA',\
                                                                                    'AMD_AREDS','Multi_Retina', 'Multi_JSIEC'])
    args = parser.parse_args()

    # load an image
    img_path = args.img_path
    img = remove_black_borders_fast(img_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)
    if args.mode == 'mae':
        chkpt_dir = './checkpoint/PreTraining/checkpoint-best.pth'
        model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
        
        # make random mask reproducible (comment out to make it change)
        torch.manual_seed(4)
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std    
        run_one_image(img, model_mae)

    elif args.mode == 'cam':
        # chose fine-tuned model from 'checkpoint'
        chkpt_dir = os.path.join('./checkpoint', args.ft_model, 'checkpoint-best.pth')
        model = prepare_ft_model(chkpt_dir)
        cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1], use_cuda=True, reshape_transform=reshape_transform)
        run_cam(img, cam)
    
    elif args.mode == 'classification':
        # chose fine-tuned model from 'checkpoint'
        chkpt_dir = os.path.join('./checkpoint', args.ft_model, 'checkpoint-best.pth')
        model = prepare_ft_model(chkpt_dir, model_type = args.ft_model)
        run_classification(img, model, args.ft_model)


