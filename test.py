import os
import torch
import torch.nn.functional as F
import tiffile as tiff
import matplotlib
from tqdm.auto import tqdm
from gen_mask import gen_mask
from losses.gms_loss import MSGMS_Score
from skimage import morphology
from skimage.segmentation import mark_boundaries
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import argparse
import matplotlib.pyplot as plt
import numpy as np
import models
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import MvtecADDataset
from PIL import Image

def denormalization(x):

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x

def plot_fig(test_img, recon_imgs, scores, gts, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(0,num,100):
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

# 테스트 시 anomaly map 저장
def test_and_save_anomaly_maps(model, test_loader, RIAD, img_size, device, root_anomaly_map_dir):
    msgms_score = MSGMS_Score()
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks, labels, original_image_size, image_paths) in enumerate(tqdm(test_loader)):
            score = 0
            images = images.to(device)
            test_imgs.extend(images.cpu().numpy())
            gt_list.extend(labels.cpu().numpy())
            gt_mask_list.extend(masks.cpu().numpy())
            if RIAD:
                for k in [2,4,8,16]:
                    N = img_size // k
                    mask_generator = gen_mask([k], 3, img_size)
                    raid_masks = next(mask_generator)
                    inputs = [images * (torch.tensor(mask, requires_grad=False).to(device)) for mask in raid_masks]
                    outputs = [model(x) for x in inputs]
                    outputs = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, raid_masks))
                    score += msgms_score(images, outputs) / (N**2)
                
            else:
                outputs = model(images)
                score = msgms_score(images, outputs)
                # score = F.mse_loss(images, outputs, reduction='none').mean(dim=1)
                
            score = score.squeeze().cpu().numpy()
            for i in range(score.shape[0]):
                score[i] = gaussian_filter(score[i], sigma=7)

            scores.extend(score)
            recon_imgs.extend(outputs.cpu().numpy())
            
            # 배치의 각 이미지에 대해 anomaly map 저장 
            for i in range(images.size(0)):
                image_path = image_paths[i]
                anomaly_map = score[i]
                save_anomaly_map(anomaly_map, image_path, root_anomaly_map_dir, img_size=img_size)
                
    return scores, test_imgs, recon_imgs, gt_list, gt_mask_list

def test(model, test_loader, root_anomaly_map_dir, RIAD, img_size, device, save_name):
    
    scores, test_imgs, recon_imgs, gt_list, gt_mask_list = test_and_save_anomaly_maps(model, test_loader, RIAD, img_size, device=device, root_anomaly_map_dir=root_anomaly_map_dir)
    
    scores = np.asarray(scores)
    # max_anomaly_score = scores.max()
    # min_anomaly_score = scores.min()
    # scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print('image ROCAUC: %.3f' % (img_roc_auc))
    plt.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % ('mvtec', img_roc_auc))
    plt.legend(loc="lower right")

    # calculate per-pixel level ROCAUC
    gt_mask = np.asarray(gt_mask_list)
    gt_mask = (gt_mask > 0.5).astype(int) # 확실히 0,1 만 갖게 만듦
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

    plt.plot(fpr, tpr, label='%s pixel_ROCAUC: %.3f' % ('mvtec', per_pixel_rocauc))
    plt.legend(loc="lower right")
    save_dir = f'metrics/{save_name}/pictures_{threshold:.4f}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=100)

    plot_fig(test_imgs, recon_imgs, scores, gt_mask_list, threshold, save_dir)
    return model

def return_model(model_name:str):
    cls = getattr(models, model_name)  
    return cls()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--state_dict_path', type=str)
    parser.add_argument('--data_path', type=str, default='mvtec_anomaly_detection')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--RIAD', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=3338)

    args = parser.parse_args()
        
    test_dataset = MvtecADDataset(root_dir=f"mvtec_anomaly_detection_{args.img_size}", split="test", img_size=args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = return_model(args.model_name).to(device)
    model.load_state_dict(torch.load(args.state_dict_path, weights_only=True))

    save_name = os.path.basename(args.state_dict_path)
    print('Save name: ', save_name)
    os.makedirs(f'metrics/{save_name}', exist_ok=True)
    
    anomaly_dir = f'anomaly_maps/{save_name}'
    model = test(model, test_loader, root_anomaly_map_dir=anomaly_dir, RIAD=args.RIAD, img_size=args.img_size, device=device, save_name=save_name)
    
