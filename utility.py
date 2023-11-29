import sys
try:
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
except:
    from skimage.measure import compare_psnr, compare_ssim

import os
import math
import time
import datetime
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def remove_black_borders_fast(image_path, tolerance=20):
    with Image.open(image_path) as img:
        # Convert the image to a numpy array
        img_array = np.array(img)
        brightness_sum = img_array.sum(axis=2)

        # Find all rows and columns where the brightness is above a certain threshold
        # Assuming black pixels have very low brightness values
        threshold = brightness_sum.max() * 0.1 # 10% of the max value
        non_black_rows = np.where(brightness_sum.max(axis=1) > threshold)[0]
        non_black_cols = np.where(brightness_sum.max(axis=0) > threshold)[0]

        # Find the bounding box of the non-black areas
        top_row = non_black_rows[0]
        bottom_row = non_black_rows[-1]
        left_col = non_black_cols[0]
        right_col = non_black_cols[-1]

        cropped_img = img.crop((left_col, top_row, right_col, bottom_row))
        return cropped_img
    
def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    # plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.imshow(image)
    plt.title(title, fontsize=64)
    plt.axis('off')
    return

def prepare_data(image):
    input_img = image - imagenet_mean
    input_img = image / imagenet_std    
    x = torch.tensor(input_img)
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x).float()
    return x

def draw_mae(image, mask, y):
    # make the plt figure larger
    fig = plt.figure(figsize=(24, 24))

    image = torch.tensor(image).unsqueeze(0)
    im_masked = image * (1 - mask)    
    y = torch.clip((y * imagenet_std + imagenet_mean) * 255, 0, 255)    
    x = image * 255
    im_masked = im_masked * 255
    im_paste = x * (1 - mask) + y * mask

    plt.subplot(2, 2, 1)
    show_image(x[0].numpy().astype(np.uint8), "original")

    plt.subplot(2, 2, 2)
    show_image(im_masked[0].numpy().astype(np.uint8), "masked")
    
    plt.subplot(2, 2, 3)
    show_image(y[0].numpy().astype(np.uint8), "reconstruction")

    plt.subplot(2, 2, 4)
    show_image(im_paste[0].numpy().astype(np.uint8), "reconstruction + visible")
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def draw_heatmap(MODEL, img, x, gpu=True):    
    cam = GradCAM(model=MODEL, target_layers=[MODEL.blocks[-1].norm1], use_cuda=gpu, reshape_transform=reshape_transform)
    target_category = None # 可以指定一个类别，或者使用 None 表示最高概率的类别
    input_tensor = x
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
    grayscale_cam = grayscale_cam[0, :]

    # 将 grad-cam 的输出叠加到原始图像上
    visualization = show_cam_on_image(img, grayscale_cam, image_weight=0.5)
    return visualization
    
def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def draw_result(probabilities, categories, colors):

    # Creating the bar plot
    fig = plt.figure(figsize=(12, 10))
    plt.barh(categories, probabilities, color=colors)
    if len(categories) == 39:
        fontsize = 8
    else:
        fontsize = 12
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Probability', fontsize=fontsize)        
    plt.ylabel('DR Category', fontsize=fontsize)        
    plt.title('Probability Distribution for Different DR Categories', fontsize=fontsize)
    plt.xlim(0, 1)  # Ensuring the x-axis ranges from 0 to 1

    # plt.show()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def to_color(arr, pmin=1, pmax=99.8, gamma=1., colors=((0, 1, 0), (1, 0, 1), (0, 1, 1))):
    """Converts a 2D or 3D stack to a colored image (maximal 3 channels).

    Parameters
    ----------
    arr : numpy.ndarray
        2D or 3D input data
    pmin : float
        lower percentile, pass -1 if no lower normalization is required
    pmax : float
        upper percentile, pass -1 if no upper normalization is required
    gamma : float
        gamma correction
    colors : list
        list of colors (r,g,b) for each channel of the input

    Returns
    -------
    numpy.ndarray
        colored image
    """
    if not arr.ndim in (2, 3):
        raise ValueError("only 2d or 3d arrays supported")
    
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    
    ind_min = np.argmin(arr.shape)
    arr = np.moveaxis(arr, ind_min, 0).astype(np.float32)
    
    out = np.zeros(arr.shape[1:] + (3,))
    
    eps = 1.e-20
    if pmin >= 0:
        mi = np.percentile(arr, pmin, axis=(1, 2), keepdims=True)
    else:
        mi = 0
    
    if pmax >= 0:
        ma = np.percentile(arr, pmax, axis=(1, 2), keepdims=True)
    else:
        ma = 1. + eps
    
    arr_norm = (1. * arr - mi) / (ma - mi + eps)
    
    for i_stack, col_stack in enumerate(colors):
        if i_stack >= len(arr):
            break
        for j, c in enumerate(col_stack):
            out[..., j] += c * arr_norm[i_stack]
    
    return np.clip(out, 0, 1)

def savecolorim(save, im, norm=True, **imshow_kwargs):
    # im: Uint8
    imshow_kwargs['cmap'] = 'magma'
    if not norm:  # 不对当前图片归一化处理，直接保存
        imshow_kwargs['vmin'] = 0
        imshow_kwargs['vmax'] = 255
    
    im = np.asarray(im)

    if save is not None:
        plt.imsave(save, im, **imshow_kwargs)
    else:
        # Make a random plot...
        fig = plt.figure()
        fig.add_subplot(111)

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        plt.imshow(im, **imshow_kwargs)
        plt.axis('off')
        fig.subplots_adjust(bottom = 0)
        fig.subplots_adjust(top = 1)
        fig.subplots_adjust(right = 1)
        fig.subplots_adjust(left = 0)

        # Now we can save it to a numpy array.
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        rp = os.path.dirname(__file__)
        
        if not args.load:
            # if not args.save:
            #     args.save = now
            self.dir = os.path.join(rp, 'experiment', args.save)
        else:
            self.dir = os.path.join(rp, 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        # os.makedirs(self.get_path('results-{}'.format(args.data_test)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 0  # 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)

        # self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        # torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)
                

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def compute_psnr_and_ssim(image1, image2, border_size=0):
    """
    Computes PSNR and SSIM index from 2 images.
    We round it and clip to 0 - 255. Then shave 'scale' pixels from each border.
    """
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)
    
    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None
    
    if border_size > 0:
        image1 = image1[border_size:-border_size, border_size:-border_size, :]
        image2 = image2[border_size:-border_size, border_size:-border_size, :]
    
    psnr = compare_psnr(image1, image2, data_range=255)
    ssim = compare_ssim(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03,
                        sigma=1.5, data_range=255)
    
    return psnr, ssim


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

from tifffile import imsave
import warnings
def save_tiff_imagej_compatible(file, img, axes, **imsave_kwargs):
    """Save image in ImageJ-compatible TIFF format.

    Parameters
    ----------
    file : str
        File name
    img : numpy.ndarray
        Image
    axes: str
        Axes of ``img``
    imsave_kwargs : dict, optional
        Keyword arguments for :func:`tifffile.imsave`

    """
    axes = axes_check_and_normalize(axes, img.ndim, disallowed='S')
    
    # convert to imagej-compatible data type
    t = img.dtype
    if 'float' in t.name:
        t_new = np.float32
    elif 'uint' in t.name:
        t_new = np.uint16 if t.itemsize >= 2 else np.uint8
    elif 'int' in t.name:
        t_new = np.int16
    else:
        t_new = t
    img = img.astype(t_new, copy=False)
    if t != t_new:
        warnings.warn("Converting data type from '%s' to ImageJ-compatible '%s'." % (t, np.dtype(t_new)))
    
    # move axes to correct positions for imagej
    img = move_image_axes(img, axes, 'TZCYX', True)
    
    imsave_kwargs['imagej'] = True
    imsave(file, img, **imsave_kwargs)
import collections

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)


def _raise(e):
    raise e


def axes_check_and_normalize(axes, length=None, disallowed=None, return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    assert axes is not None
    axes = str(axes).upper()
    consume(
        a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s." % (a, list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'." % a)) for a in axes)
    consume(axes.count(a) == 1 or _raise(ValueError("axis '%s' occurs more than once." % a)) for a in axes)
    length is None or len(axes) == length or _raise(ValueError('axes (%s) must be of length %d.' % (axes, length)))
    return (axes, allowed) if return_allowed else axes


def move_image_axes(x, fr, to, adjust_singletons=False):
    """
    x: ndarray
    fr,to: axes string (see `axes_dict`)
    """
    fr = axes_check_and_normalize(fr, length=x.ndim)
    to = axes_check_and_normalize(to)
    
    fr_initial = fr
    x_shape_initial = x.shape
    adjust_singletons = bool(adjust_singletons)
    if adjust_singletons:
        # remove axes not present in 'to'
        slices = [slice(None) for _ in x.shape]
        for i, a in enumerate(fr):
            if (a not in to) and (x.shape[i] == 1):
                # remove singleton axis
                slices[i] = 0
                fr = fr.replace(a, '')
        x = x[tuple(slices)]
        # add dummy axes present in 'to'
        for i, a in enumerate(to):
            if (a not in fr):
                # add singleton axis
                x = np.expand_dims(x, -1)
                fr += a
    
    if set(fr) != set(to):
        _adjusted = '(adjusted to %s and %s) ' % (x.shape, fr) if adjust_singletons else ''
        raise ValueError(
            'image with shape %s and axes %s %snot compatible with target axes %s.'
            % (x_shape_initial, fr_initial, _adjusted, to)
        )
    
    ax_from, ax_to = axes_dict(fr), axes_dict(to)
    if fr == to:
        return x
    return np.moveaxis(x, [ax_from[a] for a in fr], [ax_to[a] for a in fr])


def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes, return_allowed=True)
    return {a: None if axes.find(a) == -1 else axes.find(a) for a in allowed}
    # return collections.namedtuple('Axes',list(allowed))(*[None if axes.find(a) == -1 else axes.find(a) for a in allowed ])
