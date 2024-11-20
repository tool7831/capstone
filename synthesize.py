from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets.dataset import MvtecADDataset, OBJECT_NAMES
from losses.ssim_loss import SSIM_Loss
from losses.gms_loss import MSGMS_Loss, MSGMS_Score
from utils.early_stopping import EarlyStopping
from utils.save import save_anomaly_map, plot_fig, save_model
from scipy.ndimage import gaussian_filter
from torchvision.utils import save_image
from eval.evaluate_experiment import *
import numpy as np
import random
import os
import json
import torch
import torch.nn as nn
import argparse
from models.sae import SimpleSAE

class SimpleAD():
    def __init__(self, args):
        self.args = args
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        train_data = MvtecADDataset(root_dir=f"mvtec_anomaly_detection_{args.img_size}", split="train", img_size=args.img_size)
        img_nums = len(train_data)
        valid_num = int(img_nums * 0.2)
        train_num = img_nums - valid_num
        train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_num, valid_num])

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        # 모델 학습 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleSAE().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        if args.prefix is None:
            self.save_name = f'{self.model.__class__.__name__}_{args.img_size}'
        else:
            self.save_name = f'{self.model.__class__.__name__}_{args.img_size}_{args.prefix}'
        
        os.makedirs(f'metrics/{self.save_name}', exist_ok=True)
        with open(os.path.join(f'metrics/{self.save_name}', 'model_training_log.txt'), 'w') as f:
            state = {k: v for k, v in args._get_kwargs()}
            f.write(str(state))
        
        # fetch fixed data for debugging
        x_normal_fixed, _, _, _, _ = next(iter(self.valid_loader))
        self.x_normal_fixed = x_normal_fixed.to(self.device)

        test_dataset = MvtecADDataset(root_dir=f"mvtec_anomaly_detection_{args.img_size}", split="test", img_size=args.img_size)
        random_indices = random.sample(range(len(test_dataset)), args.batch_size)
        random_subset = Subset(test_dataset, random_indices)
        random_loader = DataLoader(random_subset, batch_size=args.batch_size, shuffle=False)
        x_test_fixed, _, _, _, _ = next(iter(random_loader))
        self.x_test_fixed = x_test_fixed.to(self.device)   
        
    def train(self):
        # 학습 루프
        best_loss = 100000
        early_stopping = EarlyStopping(patience=10)

        for epoch in tqdm(range(self.args.epochs)):
            
            train_loss, train_l1_loss, train_l2_loss, train_gms_loss, train_ssim_loss = self._train()
            valid_loss = self._eval()
            
            if epoch % 10 == 9:
                save_sample = os.path.join(f'metrics/{self.save_name}', f'{epoch+1}-images.jpg')
                save_sample2 = os.path.join(f'metrics/{self.save_name}', f'{epoch+1}test-images.jpg')
                self.save_snapshot(self.x_normal_fixed, self.x_test_fixed, save_sample, save_sample2)
                
            self.scheduler.step(valid_loss / len(self.valid_loader))
            early_stopping(val_loss=valid_loss / len(self.valid_loader))
            best_loss = save_model(self.model, train_loss / len(self.train_loader), valid_loss / len(self.valid_loader), best_loss, epoch+1, self.save_name)
            
            
            print(f"Epoch [{epoch+1}/{self.args.epochs}], Train Loss: {train_loss / len(self.train_loader):.4f}, Valid Loss {valid_loss / len(self.valid_loader):.4f}")
            print(f'Train L1_Loss: {train_l1_loss / len(self.train_loader) * self.args.delta:.6f} L2_Loss: {train_l2_loss / len(self.train_loader)* self.args.gamma:.6f} GMS_Loss: {train_gms_loss / len(self.train_loader)* self.args.alpha:.6f} SSIM_Loss: {train_ssim_loss / len(self.train_loader)* self.args.beta:.6f}')
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
    def _train(self):
        self.model.train()
        ssim = SSIM_Loss()
        mse = nn.MSELoss()
        msgms = MSGMS_Loss()
        l1 = nn.L1Loss()
        train_loss = 0
        train_l1_loss = 0
        train_l2_loss = 0
        train_gms_loss = 0
        train_ssim_loss = 0
        for images, _, _, _, _ in tqdm(self.train_loader):
            if torch.isnan(images).any():
                print("NaN detected in input images")
                continue  # NaN이 포함된 이미지는 건너뛰기
            
            if torch.isinf(images).any():
                print("Inf detected in input images")
                continue  # Inf가 포함된 이미지는 건너뛰기
            images = images.to(self.device)
            self.optimizer.zero_grad()
            outputs, mask = self.model.train_model(images)
            l1_loss = l1(images, outputs)
            l2_loss = mse(images, outputs)
            gms_loss = msgms(images, outputs)
            ssim_loss = ssim(images, outputs)
            loss = l1_loss * self.args.delta + self.args.gamma * l2_loss + self.args.alpha * gms_loss + self.args.beta * ssim_loss
            
            train_loss += loss.item()
            train_l1_loss += l1_loss.item()
            train_l2_loss += l2_loss.item()
            train_gms_loss += gms_loss.item()
            train_ssim_loss += ssim_loss.item()

            loss.backward()
            self.optimizer.step()
            
        return train_loss, train_l1_loss, train_l2_loss, train_gms_loss, train_ssim_loss
    
    def _eval(self):
        self.model.eval()
        ssim = SSIM_Loss()
        mse = nn.MSELoss()
        msgms = MSGMS_Loss()
        l1 = nn.L1Loss()
        valid_l1_loss = 0
        valid_l2_loss = 0
        valid_gms_loss = 0
        valid_ssim_loss = 0
        valid_loss = 0
        with torch.no_grad():
            for images, _, _, _, _ in tqdm(self.valid_loader):
                images = images.to(self.device)
                outputs, mask = self.model.train_model(images)

                l1_loss = l1(images, outputs)
                l2_loss = mse(images, outputs)
                gms_loss = msgms(images, outputs)
                ssim_loss = ssim(images, outputs)
                loss = self.args.delta * l1_loss + self.args.gamma * l2_loss + self.args.alpha * gms_loss + self.args.beta * ssim_loss

                valid_l1_loss += l1_loss.item()
                valid_l2_loss += l2_loss.item()
                valid_gms_loss += gms_loss.item()
                valid_ssim_loss += ssim_loss.item()
                valid_loss += loss.item()
        return valid_loss
          
    def _test(self, test_loader, root_anomaly_map_dir):
        msgms_score = MSGMS_Score()
        scores = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        recon_imgs = []
        self.model.eval()
        with torch.no_grad():
            for images, masks, labels, _, image_paths in tqdm(test_loader):
                score = 0
                images = images.to(self.device)
                test_imgs.extend(images.cpu().numpy())
                gt_list.extend(labels.cpu().numpy())
                gt_mask_list.extend(masks.cpu().numpy())
                outputs, mask = self.model(images)
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
                    save_anomaly_map(anomaly_map, image_path, root_anomaly_map_dir, img_size=self.args.img_size)
                    
        return scores, test_imgs, recon_imgs, gt_list, gt_mask_list
    
    def test(self, evaluated_objects, pro_integration_limit=0.3):
        
        assert 0.0 < pro_integration_limit <= 1.0
        root_anomaly_map_dir=f'anomaly_maps/{self.save_name}'
        output_dir=f'metrics/{self.save_name}'
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
            
            test_dataset = MvtecADDataset(root_dir=f"mvtec_anomaly_detection_{self.args.img_size}", split="test", img_size=self.args.img_size, object_names=[obj])
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
            scores, test_imgs, recon_imgs, gt_list, gt_mask_list = self._test(test_loader=test_loader, root_anomaly_map_dir=root_anomaly_map_dir)
            scores = np.asarray(scores)

            # Calculate the PRO and ROC curves.
            au_pro, au_roc, pro_curve, roc_curve, pixel_level_metrics, image_level_metrics = \
                calculate_metrics(
                    np.asanyarray(gt_mask_list).squeeze(axis=1),
                    scores,
                    pro_integration_limit)
                
            threshold = pixel_level_metrics['threshold']
            save_dir = f'metrics/{self.save_name}/pictures_{obj}'
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
    
    def load_model(self, state_dict_path):
        self.model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    
    def save_snapshot(self, x, x2, save_dir, save_dir2):
        self.model.eval()
        with torch.no_grad():
            x_fake_list = x
            recon, _ = self.model(x)
            x_concat = torch.cat((x_fake_list, recon), dim=3)
            save_image((x_concat.data.cpu()), save_dir, nrow=1, padding=0)
            print(('Saved real and fake images into {}...'.format(save_dir)))

            x_fake_list = x2
            recon, _ = self.model(x2)
            x_concat = torch.cat((x_fake_list, recon), dim=3)
            save_image((x_concat.data.cpu()), save_dir2, nrow=1, padding=0)
            print(('Saved real and fake images into {}...'.format(save_dir2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    args = parser.parse_args()

    exp = SimpleAD(args=args)

    exp.train()
    exp.test(evaluated_objects=OBJECT_NAMES)