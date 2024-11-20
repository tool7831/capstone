import os
import torch
import numpy as np
import tifffile as tiff
import matplotlib
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.segmentation import mark_boundaries
from torchvision.utils import save_image

def save_snapshot(x, x2, model, save_dir, save_dir2, dim=3):
    model.eval()
    with torch.no_grad():
        x_fake_list = x
        recon = model(x)
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        save_image((x_concat.data.cpu()), save_dir, nrow=1, padding=0)
        print(('Saved real and fake images into {}...'.format(save_dir)))

        x_fake_list = x2
        recon = model(x2)
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        save_image((x_concat.data.cpu()), save_dir2, nrow=1, padding=0)
        print(('Saved real and fake images into {}...'.format(save_dir2)))
        
def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_model(model, train_loss, valid_loss, best_loss, epoch, save_name):
    loss_dir =  f"metrics/{save_name}"
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    with open(loss_dir + '/loss', 'w') as f:
        f.write(f'Epoch {epoch}, Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f}\n' ,)
        
    if valid_loss < best_loss:
        if not os.path.exists(f'save'):
            os.makedirs(f'save')
        torch.save(model.state_dict(), f'save/{save_name}')
        best_loss = valid_loss
        
    return best_loss

def save_anomaly_map(anomaly_map, image_path, anomaly_root_dir, img_size):
    """
    anomaly_map을 원본 이미지와 동일한 폴더 구조로 anomaly_root_dir에 저장합니다.
    
    Args:
        anomaly_map (Tensor): anomaly map 이미지
        image_path (str): 원본 이미지 경로
        anomaly_root_dir (str): anomaly map의 최상위 폴더 경로
    """
    # 이미지의 파일 경로에서 최상위 디렉토리를 제외한 경로 추출
    relative_path = os.path.relpath(image_path, start=f'mvtec_anomaly_detection_{img_size}')
    relative_path = os.path.splitext(relative_path)[0] + '.tiff'
    save_path = os.path.join(anomaly_root_dir, relative_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # anomaly map 저장
    tiff.imwrite(save_path, anomaly_map)
    
    
def denormalization(x):
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x

def plot_fig(test_img, recon_imgs, scores, gts, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(0,num,50):
        img = test_img[i]
        img = denormalization(img)
        recon_img = recon_imgs[i]
        recon_img = denormalization(recon_img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 6, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(recon_img)
        ax_img[1].title.set_text('Reconst')
        ax_img[2].imshow(gt, cmap='gray')
        ax_img[2].title.set_text('GroundTruth')
        ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[3].title.set_text('Predicted heat map')
        ax_img[4].imshow(mask, cmap='gray')
        ax_img[4].title.set_text('Predicted mask')
        ax_img[5].imshow(vis_img)
        ax_img[5].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, f'{i}_png'), dpi=100)
        plt.close()