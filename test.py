import os
import torch
import argparse
import numpy as np

import json
from models import autoencoder
from tqdm.auto import tqdm
from utils.gen_mask import gen_mask
from losses.gms_loss import MSGMS_Score
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from datasets.dataset import MvtecADDataset, OBJECT_NAMES
from eval.evaluate_experiment import *
from utils.save import *

def return_model(model_name:str):
    cls = getattr(autoencoder, model_name)  
    return cls()

def _test(args, model, test_loader, root_anomaly_map_dir, device):
    msgms_score = MSGMS_Score()
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    model.eval()
    with torch.no_grad():
        for images, masks, labels, _, image_paths in tqdm(test_loader):
            score = 0
            images = images.to(device)
            test_imgs.extend(images.cpu().numpy())
            gt_list.extend(labels.cpu().numpy())
            gt_mask_list.extend(masks.cpu().numpy())
            if args.RIAD:
                for k in [2,4,8,16]:
                    N = args.img_size // k
                    mask_generator = gen_mask([k], 3, args.img_size)
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
                save_anomaly_map(anomaly_map, image_path, root_anomaly_map_dir, img_size=args.img_size)
                
    return scores, test_imgs, recon_imgs, gt_list, gt_mask_list

def test(args, model, device, save_name, evaluated_objects,  pro_integration_limit=0.3):
    
    assert 0.0 < pro_integration_limit <= 1.0
    root_anomaly_map_dir=f'anomaly_maps/{save_name}'
    output_dir=f'metrics/{save_name}'
    evaluation_dict = dict()
    # Keep track of the mean performance measures.
    au_pros = []
    au_rocs = []
    
    p_acs = []
    p_prs = []
    p_res = []
    p_f1s = []
    i_acs = []
    i_prs = []
    i_res = []
    i_f1s = []
    
    # Evaluate each dataset object separately.
    for obj in evaluated_objects:
        print(f"=== Evaluate {obj} ===")
        evaluation_dict[obj] = dict()
        
        test_dataset = MvtecADDataset(root_dir=f"mvtec_anomaly_detection_{args.img_size}", split="test", img_size=args.img_size, object_names=[obj])
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        scores, test_imgs, recon_imgs, gt_list, gt_mask_list = _test(args=args, model=model, test_loader=test_loader, root_anomaly_map_dir=root_anomaly_map_dir, device=device, )
        scores = np.asarray(scores)

        # Calculate the PRO and ROC curves.
        au_pro, au_roc, pro_curve, roc_curve, pixel_level_metrics, image_level_metrics = \
            calculate_metrics(
                np.asanyarray(gt_mask_list).squeeze(axis=1),
                scores,
                pro_integration_limit)
            
        threshold = pixel_level_metrics['threshold']
        save_dir = f'metrics/{save_name}/pictures_{obj}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_img=test_imgs, recon_imgs=recon_imgs, scores=scores, gts=gt_mask_list, threshold=threshold, save_dir=save_dir)
        
        evaluation_dict[obj]['au_pro'] = au_pro
        evaluation_dict[obj]['classification_au_roc'] = au_roc
        evaluation_dict[obj]['pixel_level_accuracy'] = pixel_level_metrics['accuracy']
        evaluation_dict[obj]['pixel_level_precision'] = pixel_level_metrics['precision']
        evaluation_dict[obj]['pixel_level_recall'] = pixel_level_metrics['recall']
        evaluation_dict[obj]['pixel_level_f1_score'] = pixel_level_metrics['f1']
        evaluation_dict[obj]['image_level_accuracy'] = image_level_metrics['accuracy']
        evaluation_dict[obj]['image_level_precision'] = image_level_metrics['precision']
        evaluation_dict[obj]['image_level_recall'] = image_level_metrics['recall']
        evaluation_dict[obj]['image_level_f1_score'] = image_level_metrics['f1']
        

        evaluation_dict[obj]['classification_roc_curve_fpr'] = roc_curve[0]
        evaluation_dict[obj]['classification_roc_curve_tpr'] = roc_curve[1]

        # Keep track of the mean performance measures.
        au_pros.append(au_pro)
        au_rocs.append(au_roc)
        p_acs.append(pixel_level_metrics['accuracy'])
        p_prs.append(pixel_level_metrics['precision'])
        p_res.append(pixel_level_metrics['recall'])
        p_f1s.append(pixel_level_metrics['f1'])
        i_acs.append(image_level_metrics['accuracy'])
        i_prs.append(image_level_metrics['precision'])
        i_res.append(image_level_metrics['recall'])
        i_f1s.append(image_level_metrics['f1'])

        print('\n')

    # Compute the mean of the performance measures.
    evaluation_dict['mean_au_pro'] = np.mean(au_pros).item()
    evaluation_dict['mean_classification_au_roc'] = np.mean(au_rocs).item()
    
    evaluation_dict['mean_pixel_level_accuracy'] = np.mean(p_acs).item()
    evaluation_dict['mean_pixel_level_precision'] = np.mean(p_prs).item()
    evaluation_dict['mean_pixel_level_recall'] = np.mean(p_res).item()
    evaluation_dict['mean_pixel_level_f1_score'] = np.mean(p_f1s).item()
    evaluation_dict['mean_image_level_accuracy'] = np.mean(i_acs).item()
    evaluation_dict['mean_image_level_precision'] = np.mean(i_prs).item()
    evaluation_dict['mean_image_level_recall'] = np.mean(i_res).item()
    evaluation_dict['mean_image_level_f1_score'] = np.mean(i_f1s).item()

    # If required, write evaluation metrics to drive.
    if output_dir is not None:
        makedirs(output_dir, exist_ok=True)

        with open(path.join(output_dir, 'metrics.json'), 'w') as file:
            json.dump(evaluation_dict, file, indent=4)

        print(f"Wrote metrics to {path.join(output_dir, 'metrics.json')}")

    return model

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
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = return_model(args.model_name).to(device)
    model.load_state_dict(torch.load(args.state_dict_path, weights_only=True))

    save_name = os.path.basename(args.state_dict_path)
    print('Save name: ', save_name)
    os.makedirs(f'metrics/{save_name}', exist_ok=True)
    
    model = test(args, model, device=device, save_name=save_name, evaluated_objects=OBJECT_NAMES)
