import os
import torch
import argparse
import random
import train, test
from models import autoencoder
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets.dataset import MvtecADDataset, OBJECT_NAMES
    
def return_model(model_name:str):
    cls = getattr(autoencoder, model_name)  
    return cls()
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', type=str )
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--RIAD', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    args = parser.parse_args()
    
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = MvtecADDataset(root_dir=f"mvtec_anomaly_detection_{args.img_size}", split="test", img_size=args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 모델 학습 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = return_model(args.model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    if args.RIAD:
        if args.prefix is None:
            save_name = f'{model.__class__.__name__}_RIAD_{args.img_size}'
        else:
            save_name = f'{model.__class__.__name__}_RIAD_{args.img_size}_{args.prefix}'
    else:
        if args.prefix is None:
            save_name = f'{model.__class__.__name__}_{args.img_size}'
        else:
            save_name = f'{model.__class__.__name__}_{args.img_size}_{args.prefix}'
        
    os.makedirs(f'metrics/{save_name}', exist_ok=True)
    with open(os.path.join(f'metrics/{save_name}', 'model_training_log.txt'), 'w') as f:
        state = {k: v for k, v in args._get_kwargs()}
        f.write(str(state))
    
    # fetch fixed data for debugging
    x_normal_fixed, _, _, _, _ = next(iter(valid_loader))
    x_normal_fixed = x_normal_fixed.to(device)

    random_indices = random.sample(range(len(test_dataset)), args.batch_size)
    random_subset = Subset(test_dataset, random_indices)
    random_loader = DataLoader(random_subset, batch_size=args.batch_size, shuffle=False)
    x_test_fixed, _, _, _, _ = next(iter(random_loader))
    x_test_fixed = x_test_fixed.to(device)
    
        
    model = train.train(args, model, train_loader, valid_loader, optimizer, scheduler, device=device, save_name=save_name, x_normal_fixed=x_normal_fixed, x_test_fixed=x_test_fixed)
    model = test.test(args, model, device=device, save_name=save_name, evaluated_objects=OBJECT_NAMES)

   
    
    
