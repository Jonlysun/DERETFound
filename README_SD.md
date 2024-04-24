# Controllable Generative Model Enables Ultra-High Data Efficiency for Building Generalist Medical Foundation Model

We present the information and deployment of the retinal image stable diffusion model v1.4 (ReSDv1.4) used in RETFound-DE here. 

## Bassic Information:
- Retinal image stable diffusion model v1.4 (ReSDv1.4) is based on stable diffusion v1.4. 
- We fine-tuned it for 60000 iteration on 150k retinal image text-image pairs. 
- It follows text-to-image stragety to generate retinal image and takes about 7 seconds to generate an image each time on an NVIDIA GTX 3090.
- The resolution of generated retinal image is 512x512.

## Prepare the environment

1. Download the stable diffusion v1.4 and ReSDv1.4 model

You can download the stable diffusion v1.4 from [HuggingFace](https://huggingface.co/CompVis/stable-diffusion-v1-4) and ReSDv1.4 from [Zenodo:sd-retina-model](https://zenodo.org/records/10947092) or [baiduDisk code:7n7v](https://pan.baidu.com/s/1TBVNlaR9xW_rqA8ZdrRuOg).


2. Install Diffusers
   
Install [Diffusers](https://github.com/huggingface/diffusers) in a virtual environment from PyPI or Conda.

Please note that the version of Diffusers may influence the deployment. In our experiments, we use Diffusers v0.21.4.

## Inference

```
    import os
    from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel
    
    my_model_path = "ReSDv1.4 path, e.g., /home/user/sd-retinal-model/checkpoint-60000/ " 
    pre_trained_model = "stable diffusion v1.4 path, e.g., /home/user/stable-diffusion-v1-4 "
    text_encoder = CLIPTextModel.from_pretrained(pre_trained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pre_trained_model, subfolder="vae")

    unet = UNet2DConditionModel.from_pretrained(my_model_path, subfolder="unet")

    pipe = StableDiffusionPipeline.from_pretrained(
        pre_trained_model,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
    )

    pipe.to("cuda")

    # define your prompt
    prompt = "No Diabetic Retinopathy"

    image = pipe(prompt=prompt).images[0]    
    image.save("test.png")
```

We present some disease prompt here, for more information please refer to our paper:
```
# disease prompts for retinal image generation

Normal fundus
No referable glaucoma 
Referable glaucoma
No Diabetic Retinopathy
Diabetic Retinopathy
Mild Non-Proliferative Diabetic Retinopathy
Moderate Non-Proliferative Diabetic Retinopathy
Severe Non-Proliferative Diabetic Retinopathy
Proliferative Diabetic Retinopathy
```










Please contact 	**sunyuqi387@gmail.com** if you have questions.
