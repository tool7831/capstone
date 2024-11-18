import os
import random
import torch
import torch.nn as nn
from losses.ssim_loss import SSIM_Loss
from losses.gms_loss import MSGMS_Loss
from tqdm.auto import tqdm
from utils.gen_mask import gen_mask
from torchvision.utils import save_image

def train(model, train_loader, valid_loader, args, optimizer, scheduler, device, save_name, x_normal_fixed, x_test_fixed):
    # 학습 루프
    best_loss = 100000
    early_stopping = EarlyStopping(patience=10)
 
    ssim = SSIM_Loss()
    mse = nn.MSELoss(reduction='mean')
    msgms = MSGMS_Loss()
    l1 = nn.L1Loss()

    for epoch in tqdm(range(args.epochs)):
        # train
        model.train()
        train_loss = 0
        train_l1_loss = 0
        train_l2_loss = 0
        train_gms_loss = 0
        train_ssim_loss = 0
        for images, _, _, _, _ in tqdm(train_loader):
            if torch.isnan(images).any():
                print("NaN detected in input images")
                continue  # NaN이 포함된 이미지는 건너뛰기
            
            if torch.isinf(images).any():
                print("Inf detected in input images")
                continue  # Inf가 포함된 이미지는 건너뛰기
            
            images = images.to(device)
            optimizer.zero_grad()
            if args.RIAD:
                k_value = random.sample([2,4,8,16],1)
                mask_generator = gen_mask(k_value, 3, args.img_size)
                masks = next(mask_generator)
                inputs = [images * (torch.tensor(mask, requires_grad=False).to(device)) for mask in masks]
                
                outputs = [model(input) for input in inputs]
                outputs = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, masks))
            else:
                outputs = model(images)
            
            l1_loss = l1(images, outputs)
            l2_loss = mse(images, outputs)
            gms_loss = msgms(images, outputs)
            ssim_loss = ssim(images, outputs)
            loss = l1_loss * args.delta + args.gamma * l2_loss + args.alpha * gms_loss + args.beta * ssim_loss
            
            train_loss += loss.item()
            train_l1_loss += l1_loss.item()
            train_l2_loss += l2_loss.item()
            train_gms_loss += gms_loss.item()
            train_ssim_loss += ssim_loss.item()

            loss.backward()
            optimizer.step()
            
        # valid
        model.eval()
        valid_l1_loss = 0
        valid_l2_loss = 0
        valid_gms_loss = 0
        valid_ssim_loss = 0
        valid_loss = 0
        with torch.no_grad():
            for images, _, _, _, _ in tqdm(valid_loader):
                images = images.to(device)
                if args.RIAD:
                    k_value = random.sample([2,4,8,16],1)
                    mask_generator = gen_mask(k_value, 3, args.img_size)
                    masks = next(mask_generator)
                    inputs = [images * (torch.tensor(mask, requires_grad=False).to(device)) for mask in masks]
                    
                    outputs = [model(input) for input in inputs]
                    outputs = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, masks))
                else:
                    outputs = model(images)

                l1_loss = l1(images, outputs)
                l2_loss = mse(images, outputs)
                gms_loss = msgms(images, outputs)
                ssim_loss = ssim(images, outputs)
                loss = args.delta * l1_loss + args.gamma * l2_loss + args.alpha * gms_loss + args.beta * ssim_loss

                valid_l1_loss += l1_loss.item()
                valid_l2_loss += l2_loss.item()
                valid_gms_loss += gms_loss.item()
                valid_ssim_loss += ssim_loss.item()
                valid_loss += loss.item()
                
        if epoch % 10 == 9:
            save_sample = os.path.join(f'metrics/{save_name}', f'{epoch+1}-images.jpg')
            save_sample2 = os.path.join(f'metrics/{save_name}', f'{epoch+1}test-images.jpg')
            save_snapshot(x_normal_fixed, x_test_fixed, model, save_sample, save_sample2)
            
        scheduler.step(valid_loss / len(valid_loader))
        early_stopping(val_loss=valid_loss / len(valid_loader))
        best_loss = save(model, train_loss / len(train_loader), valid_loss / len(valid_loader), best_loss, epoch+1, save_name)
        
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Valid Loss {valid_loss / len(valid_loader):.4f}")
        print(f'Train L1_Loss: {train_l1_loss / len(train_loader) * args.delta:.6f} L2_Loss: {train_l2_loss / len(train_loader)* args.gamma:.6f} GMS_Loss: {train_gms_loss / len(train_loader)* args.alpha:.6f} SSIM_Loss: {train_ssim_loss / len(train_loader)* args.beta:.6f}')
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return model


def save(model, train_loss, valid_loss, best_loss, epoch, save_name):
    loss_dir =  f"metrics/{save_name}"
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    with open(loss_dir + '/loss', 'a+') as f:
        f.write(f'Epoch {epoch}, Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f}\n' ,)
        
    if valid_loss < best_loss:
        if not os.path.exists(f'save/{save_name}'):
            os.makedirs(f'save/{save_name}')
        torch.save(model.state_dict(), f'save/{save_name}')
        best_loss = valid_loss
        
    return best_loss
    
    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement before stopping.
            min_delta (float): Minimum improvement to qualify as a new best.
            verbose (bool): Whether to print information about early stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print("Initial best score set.")

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print("Improvement detected, resetting counter.")
                
                
def save_snapshot(x, x2, model, save_dir, save_dir2):
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
