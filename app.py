import cv2
import gradio as gr
import numpy as np
import random
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import models_mae
import models_vit
from tifffile import imread, imsave

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import utility

DEVICES = ['CPU','CUDA']
QUANT = ['float32','float16']
TASKS = ['DR_APTOS2019','DR_IDRID','DR_MESSIDOR2','Glaucoma_PAPILA','Glaucoma_Glaucoma_Fundus','Glaucoma_ORIGA','AMD_AREDS','Multi_Retina', 'Multi_JSIEC']
INPUTS = ['DR', 'Glaucoma', 'AMD', 'Multi']
MODEL = None
PreTrainingModel = None
ARGS = None

class Args:
    gpu = True
    

def load_model(type, device, progress=gr.Progress()):
    global MODEL, ARGS

    ARGS = Args()

    if device == 'CPU':
        ARGS.gpu = False
    elif device == 'CUDA':
        ARGS.gpu = True
    else:
        gr.Error("Device not found!")
        return "Device not found"

    ARGS.device = torch.device('cpu' if not ARGS.gpu else 'cuda')
        
    if 'DR' in type:
        ARGS.task = 1

        if type == 'DR_APTOS2019':
            ARGS.nb_classes = 5
            print(ARGS.nb_classes)
            ARGS.save = 'DR_APTOS2019'
            ARGS.modelpath = './checkpoint/DR_APTOS2019/checkpoint-best.pth'
        elif type == 'DR_IDRID':
            ARGS.nb_classes = 5
            print(ARGS.nb_classes)
            ARGS.save = 'DR_IDRID'
            ARGS.modelpath = './checkpoint/DR_IDRID/checkpoint-best.pth'
        elif type == 'DR_MESSIDOR2':
            ARGS.nb_classes = 5
            print(ARGS.nb_classes)
            ARGS.save = 'DR_MESSIDOR2'
            ARGS.modelpath = './checkpoint/DR_MESSIDOR2/checkpoint-best.pth'
        else:
            gr.Error("Model not found!")
            return "Model not found"
        
    elif 'Glaucoma' in type:
        ARGS.task = 2
        ARGS.patch_size = 64
        ARGS.scale = '1'
        
        if type == 'Glaucoma_PAPILA':
            ARGS.nb_classes = 3
            print(ARGS.nb_classes)                
            ARGS.save = 'Glaucoma_PAPILA'
            ARGS.modelpath = './checkpoint/Glaucoma_PAPILA/checkpoint-best.pth'
        elif type == 'Glaucoma_Glaucoma_Fundus':
            ARGS.nb_classes = 3
            print(ARGS.nb_classes)
            ARGS.save = 'Glaucoma_Glaucoma_Fundus'
            ARGS.modelpath = './checkpoint/Glaucoma_Glaucoma_Fundus/checkpoint-best.pth'
        elif type == 'Glaucoma_ORIGA':
            ARGS.nb_classes = 2
            print(ARGS.nb_classes)
            ARGS.save = 'Glaucoma_ORIGA'
            ARGS.modelpath = './checkpoint/Glaucoma_ORIGA/checkpoint-best.pth'
        else:
            gr.Error("Model not found!")
            return "Model not found"

    elif 'AMD' in type:
        ARGS.task = 3

        ARGS.nb_classes = 4
        print(ARGS.nb_classes)
        ARGS.save = 'AMD_AREDS'                
        ARGS.modelpath = './checkpoint/AMD_AREDS/checkpoint-best.pth'

    elif 'Multi' in type:
        ARGS.task = 4
        if type == 'Multi_Retina':    
            ARGS.nb_classes = 4            
            print(ARGS.nb_classes)
            ARGS.save = 'Multi_Retina'
            ARGS.modelpath = './checkpoint/Multi_Retina/checkpoint-best.pth'
        elif type == 'Multi_JSIEC':
            ARGS.nb_classes = 39
            print(ARGS.nb_classes)
            ARGS.save = 'Multi_JSIEC'
            ARGS.modelpath = './checkpoint/Multi_JSIEC/checkpoint-best.pth'
    
    if MODEL is not None:
        del MODEL
    
    print("Resume checkpoint %s" % ARGS.modelpath)
    MODEL = models_vit.__dict__['vit_large_patch16'](
        num_classes=ARGS.nb_classes,
        drop_path_rate=0.1,
        global_pool=True,
    )
    checkpoint = torch.load(ARGS.modelpath, map_location='cpu')
    MODEL.load_state_dict(checkpoint['model'], strict=False)

    
    MODEL.to(ARGS.device)
    MODEL.eval()
    return '%s Model loaded on %s'%(type, device)

def visualize(img_input, progress=gr.Progress()):
    print(f'Opening {img_input.name}...')
    ext = os.path.basename(img_input.name).split('.')[-1]

    if ext in ['png', 'jpg', 'jpeg', 'JPG']:
        image = utility.remove_black_borders_fast(img_input.name)
        image = image.resize((224, 224))
        image = np.array(image)
    else:
        gr.Error("Image must has be .png, .jpg, .jpeg, .JPG")
        return None
        
    print(f'Image shape: {image.shape}')

    if image.shape == (224, 224, 3):     
        image = utility.savecolorim(None, image, norm=True)           
        return [[image], f'2D image loaded with shape {image.shape}']    
    else:
        gr.Error("Image must has be CFP with (224, 224, 3)")
        return None
    
def rearrange3d_fn(image):
    """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
    """

    image = np.squeeze(image)  # remove channels dimension
    # print('reshape : ' + str(image.shape))
    depth, height, width = image.shape
    image_re = np.zeros([height, width, depth])
    for d in range(depth):
        image_re[:, :, d] = image[d, :, :]
    return image_re

def lf_extract_fn(lf2d, n_num=11, mode='toChannel', padding=False):
    """
    Extract different views from a single LF projection

    Params:
        -lf2d: numpy.array, 2-D light field projection in shape of [height, width, channels=1]
        -mode - 'toDepth' -- extract views to depth dimension (output format [depth=multi-slices, h, w, c=1])
                'toChannel' -- extract views to channel dimension (output format [h, w, c=multi-slices])
        -padding -   True : keep extracted views the same size as lf2d by padding zeros between valid pixels
                        False : shrink size of extracted views to (lf2d.shape / Nnum);
    Returns:
        ndarray [height, width, channels=n_num^2] if mode is 'toChannel'
                or [depth=n_num^2, height, width, channels=1] if mode is 'toDepth'
    """
    n = n_num
    h, w, c = lf2d.shape
    if padding:
        if mode == 'toDepth':
            lf_extra = np.zeros([n * n, h, w, c])  # [depth, h, w, c]
        
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, i: h: n, j: w: n, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        elif mode == 'toChannel':
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([h, w, n * n])
            
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[i: h: n, j: w: n, d] = lf2d[i: h: n, j: w: n]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)
    else:
        new_h = int(np.ceil(h / n))
        new_w = int(np.ceil(w / n))
    
        if mode == 'toChannel':
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([new_h, new_w, n * n])
        
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[:, :, d] = lf2d[i: h: n, j: w: n]
                    d += 1
        elif mode == 'toDepth':
            lf_extra = np.zeros([n * n, new_h, new_w, c])  # [depth, h, w, c]
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, :, :, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)

    return lf_extra

def _load_imgs(img_file, t2d=True):
    def normalize(x):
        max_ = np.max(x) * 1.1
        x = x / (max_ / 2.)
        x = x - 1
        return x
    
    if t2d:
        image = imageio.imread(img_file)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # uint8 0~48 (176,176,1) (649, 649,1)
        img = normalize(image)  # float64 -1~1 (176,176,1)
        img = lf_extract_fn(img, n_num=11, padding=False)  # (16, 16, 121) (59, 59, 121)
    else:
        image = imageio.volread(img_file)  # uint8 0~132  [61,176,176]
        img = normalize(image)  # float64 -1~1 (61,176,176)
        img = rearrange3d_fn(img)  # (176,176,61)

    img = img.astype(np.float32, casting='unsafe')
    # print('\r%s : %s' % (img_file, str(img.shape)), end='')
    return img

@torch.no_grad()
def run_model(img_input, type, progress=gr.Progress()):
    global MODEL, ARGS
    
    if MODEL is None:
        gr.Error("Model not loaded!")
        return [None, None]

    if img_input is None:
        gr.Error("Image not loaded!")
        return [None, None]  
    
    print(f'Opening {img_input.name}...')
    ext = os.path.basename(img_input.name).split('.')[-1]
    if ext in ['png', 'jpg', 'jpeg', 'JPG']:
        image = utility.remove_black_borders_fast(img_input.name)
        image = image.resize((224, 224))
        image = np.array(image)
    else:
        gr.Error("Image must has be .png, .jpg, .jpeg, .JPG")
        return None
        
    image = np.array(image) / 255.
    x = utility.prepare_data(image)
    x = x.to(ARGS.device)

    # model inference
    with torch.no_grad():
        output = MODEL(x)
    output = nn.Softmax(dim=1)(output)
    output = output.squeeze(0).cpu().detach().numpy()        
    
    if 'DR' in type:        
        # visualization
        categories = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        prob_result = utility.draw_result(output, categories, colors)
        return [prob_result]

    elif 'Glaucoma_PAPILA' in type:
        # visualization
        categories = ['No glaucoma', 'Suspected glaucoma', 'Glaucoma']
        colors = ['blue', 'green', 'red']
        prob_result = utility.draw_result(output, categories, colors)
        return [prob_result]

    elif 'Glaucoma_Glaucoma_Fundus' in type:
        # visualization
        categories = ['No glaucoma', 'Early glaucoma', 'Advanced glaucoma']
        colors = ['blue', 'green', 'red']
        prob_result = utility.draw_result(output, categories, colors)
        return [prob_result]

    elif 'Glaucoma_ORIGA' in type:
        # visualization
        categories = ['No glaucoma', 'Glaucoma']
        colors = ['blue', 'red']
        prob_result = utility.draw_result(output, categories, colors)
        return [prob_result]
    
    elif 'AMD_AREDS' in type:
        # visualization
        categories = ['Non AMD', 'Mild AMD', 'Moderate AMD', 'Advanced AMD']
        colors =  ['blue', 'green', 'red', 'orange']
        prob_result = utility.draw_result(output, categories, colors)
        return [prob_result]

    elif 'Multi_Retina' in type:
        # visualization
        categories = ['Normal', 'Cataract', 'Glaucoma', 'Others']
        colors =  ['blue', 'green', 'red', 'orange']
        prob_result = utility.draw_result(output, categories, colors)
        return [prob_result]

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
        return [prob_result]

    else:
        gr.Error("This task is not supported yet!")
        return [None, None]

def run_cam(img_input, type, progress=gr.Progress()):
    global MODEL, ARGS
    
    if MODEL is None:
        gr.Error("Model not loaded!")
        return [None, None]

    if img_input is None:
        gr.Error("Image not loaded!")
        return [None, None]  
    
    print(f'Opening {img_input.name}...')
    ext = os.path.basename(img_input.name).split('.')[-1]
    if ext in ['png', 'jpg', 'jpeg', 'JPG']:
        image = utility.remove_black_borders_fast(img_input.name)
        image = image.resize((224, 224))
        image = np.array(image)
    else:
        gr.Error("Image must has be .png, .jpg, .jpeg, .JPG")
        return None
        
    image = np.array(image) / 255.

    x = utility.prepare_data(image)
    x = x.to(ARGS.device)
  
    cam_result = utility.draw_heatmap(MODEL, image, x, ARGS.gpu)        
    return [cam_result]

def run_mae(img_input, device, progress=gr.Progress()):
    global MODEL, ARGS, PreTrainingModel

    ARGS = Args()
    
    if device == 'CPU':
        ARGS.gpu = False
    elif device == 'CUDA':
        ARGS.gpu = True
    else:
        gr.Error("Device not found!")
        return "Device not found"

    ARGS.device = torch.device('cpu' if not ARGS.gpu else 'cuda')
    ARGS.pretraining_model_path = './checkpoint/PreTraining/checkpoint-best.pth'

    # build model
    PreTrainingModel = getattr(models_mae, 'mae_vit_large_patch16')()
    # load model
    checkpoint = torch.load(ARGS.pretraining_model_path, map_location='cpu')
    msg = PreTrainingModel.load_state_dict(checkpoint['model'], strict=False)
    PreTrainingModel.to(ARGS.device)
    PreTrainingModel.eval()
    
    if img_input is None:
        gr.Error("Image not loaded!")
        return [None, None]  
    
    print(f'Opening {img_input.name}...')
    ext = os.path.basename(img_input.name).split('.')[-1]
    if ext in ['png', 'jpg', 'jpeg', 'JPG']:
        image = utility.remove_black_borders_fast(img_input.name)
        image = image.resize((224, 224))
        image = np.array(image)
    else:
        gr.Error("Image must has be .png, .jpg, .jpeg, .JPG")
        return None
        
    image = np.array(image) / 255.
    x = utility.prepare_data(image)
    x = x.to(ARGS.device)

    # run MAE
    loss, y, mask = PreTrainingModel(x.float(), mask_ratio=0.75)
    y = PreTrainingModel.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, PreTrainingModel.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = PreTrainingModel.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', x).detach().cpu()

    # im_paste = x * (1 - mask) + y * mask
    # x = torch.tensor(image).unsqueeze(0)
    # im_masked = x * (1 - mask)
    
    # mae_result = utility.draw_mae(x, im_masked, y, im_paste)
    mae_result = utility.draw_mae(image, mask, y)
    return [mae_result]

with gr.Blocks() as demo:

    gr.Markdown("# Online Demo for DERETFound")
    gr.Markdown("This demo allows you to run the models on your own images. Given an image and chose eye detection task, we show the  \
                 **MAE reconstructed images**,  **diagnostic probability** and **interpretable heatmaps**. Please refer to the paper for more details.")
    gr.Markdown("## Instructions")
    gr.Markdown("1. Upload your image or use the examples below. We accept colour fundus photography images.")
    gr.Markdown("2. Click 'Check Input' to inspect your input image.")    
    gr.Markdown("3. If you want to see the MAE reconstructed image by our pre-training model, please click 'Run' in 'Load and Run Pre-Training model'. This may take a while to display the image.")    
    gr.Markdown("4. If you want to see the diagnostic probability and interpretable heatmaps, select the model type in 'Load and Run Fine-tuning Model. We provide models for different tasks and datasets, including:")
    gr.Markdown("$\qquad$- diabetic retinopathy grading (Kaggle APTOS-2019, IDRiD, MESSIDOR2)") 
    gr.Markdown("$\qquad$- glaucoma diagnosis (PAPILA, Glaucoma Fundus, ORIGA)") 
    gr.Markdown("$\qquad$- Age-related macular degeneration grading (AREDS)") 
    gr.Markdown("$\qquad$- Multi-diseases classification (Retina, JSIEC)")                 
    gr.Markdown("5. Click 'Load Model' to load the model. This may take a while.")
    gr.Markdown("6. Click 'Run' to run the model on the input image. Some tasks like denoising will take several minutes to run.")
    
    gr.Markdown("Internet Content Provider ID: [沪ICP备2023024810号-1](https://beian.miit.gov.cn/)", rtl=True)

    with gr.Column():
        # with gr.Column():
        gr.Markdown("## Upload Image")
            
        with gr.Column():
            img_input = gr.File(label="Input File", interactive=True)
            img_visual = gr.Gallery(label="Input Viusalization", interactive=False)

            with gr.Row():
                load_image = gr.Textbox(label="Image Information", value="Image not loaded")
                check_input = gr.Button("Check Input") 

        with gr.Column():
            with gr.Row():
                gr.Examples(
                    label='Diabetic Retinopathy Examples',
                    examples=[
                        ["exampledata/DR/APTOS2019.png",'DR'],
                        ["exampledata/DR/IDRiD.jpg",'DR'],
                        ["exampledata/DR/MESSIDOR2-01.png",'DR'],
                        ["exampledata/DR/MESSIDOR2-02.png",'DR'],
                    ],
                    inputs=[img_input, load_image],
                )

                gr.Examples(
                    label='Glaucoma Examples',
                    examples=[
                        ["exampledata/Glaucoma/PAPILA-01.jpg",'Glaucoma'],
                        ["exampledata/Glaucoma/PAPILA-02.jpg",'Glaucoma'],
                        ["exampledata/Glaucoma/GlaucomaFundus.png",'Glaucoma'],
                        ["exampledata/Glaucoma/ORIGA.jpg",'Glaucoma'],
                    ],
                    inputs=[img_input, load_image],
                )

            with gr.Row():
                gr.Examples(
                    label='Age-related Macular Degeneration Examples',
                    examples=[
                        ["exampledata/AMD/AREDS-01.jpg","AMD"],
                        ["exampledata/AMD/AREDS-02.jpg","AMD"],
                        ["exampledata/AMD/AREDS-03.jpg","AMD"],
                        ["exampledata/AMD/AREDS-04.jpg","AMD"],
                    ],
                    inputs=[img_input, load_image],
                )
                                    
                gr.Examples(
                    label='Multi-disease Examples',
                    examples=[
                        ["exampledata/Multi-disease/JSIEC-01.JPG","multi"],
                        ["exampledata/Multi-disease/JSIEC-02.JPG","multi"],
                        ["exampledata/Multi-disease/Retina-01.png","multi"],
                        ["exampledata/Multi-disease/Retina-02.png","multi"],
                    ],
                    inputs=[img_input, load_image],
                )

    with gr.Column():
        gr.Markdown("## Load and Run Pre-Training Model")
        # output_file = gr.File(label="Output File", interactive=False)
        img_mae = gr.Gallery(label="MAE Reconstructed image")
        
        with gr.Row():
            device_1 = gr.Dropdown(label="Device", choices=DEVICES, value="CUDA")
            
        with gr.Row():
            load_progress = gr.Textbox(label="Model Information", value="DERETFound")
            mae_run_btn = gr.Button("Run")

        gr.Markdown("## Load and Run Fine-tuning Model")
        img_output = gr.Gallery(label="Classification  probability")
        img_heatmaps = gr.Gallery(label="Interpretable heatmaps")

        with gr.Row():
            device_2 = gr.Dropdown(label="Device", choices=DEVICES, value="CUDA")
        
        with gr.Row():
            type = gr.Dropdown(label="Model Type", choices=TASKS, value="DR_APTOS2019")
            load_btn = gr.Button("Load Model")
            
        with gr.Row():
            load_progress = gr.Textbox(label="Model Information", value="Model not loaded")
            run_btn = gr.Button("Run")

    check_input.click(visualize, inputs=img_input, outputs=[img_visual,load_image], queue=True)
    mae_run_btn.click(run_mae, inputs=[img_input, device_1], outputs=[img_mae], queue=True)
    load_btn.click(load_model,inputs=[type, device_2], outputs=load_progress, queue=True)
    run_btn.click(run_model, inputs=[img_input, type], outputs=[img_output], queue=True)
    run_btn.click(run_cam, inputs=[img_input, type], outputs=[img_heatmaps], queue=True)

demo.queue().launch(server_name='0.0.0.0', server_port=7891)
