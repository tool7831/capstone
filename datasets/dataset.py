import os
import torch
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
OBJECT_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
class MvtecADDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224, object_names=OBJECT_NAMES):
        self.root_dir = root_dir
        self.split = split

        # 데이터 전처리 및 데이터 로더 설정
        transform_x = transforms.Compose([
            transforms.Resize(img_size, Image.LANCZOS),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_mask = transforms.Compose([
            transforms.Resize(img_size, Image.NEAREST),
            transforms.ToTensor()
        ])
        
        self.transform_x = transform_x
        self.transform_mask = transform_mask
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        # self.images = []
        # self.masks = []
        
        for object_type in object_names:
            object_path = os.path.join(root_dir, object_type)
            if not os.path.isdir(object_path):
                continue
            
            if split == "train":
                train_dir = os.path.join(object_path, "train", "good")
                self.image_paths.extend([os.path.join(train_dir, img) for img in os.listdir(train_dir)])
                # self.images.extend([Image.open(os.path.join(train_dir, img)).convert("RGB") for img in os.listdir(train_dir)])
                self.labels.extend([0] * len(self.image_paths))
                self.mask_paths.extend([None] * len(self.image_paths))
                # self.masks.extend([None] * len(self.image_paths))

            elif split == "test":
                test_dir = os.path.join(object_path, "test")
                ground_truth_dir = os.path.join(object_path, "ground_truth")
                
                for defect_type in os.listdir(test_dir):
                    defect_dir = os.path.join(test_dir, defect_type)
                    for img_name in os.listdir(defect_dir):
                        img_path = os.path.join(defect_dir, img_name)
                        self.image_paths.append(img_path)
                        # self.images.append(Image.open(img_path).convert("RGB"))
                        
                        if defect_type == "good":
                            self.labels.append(0)
                            self.mask_paths.append(None)
                            # self.masks.append(None)
                        else:
                            self.labels.append(1)
                            mask_path = os.path.join(ground_truth_dir, defect_type, os.path.splitext(img_name)[0]+'_mask.png')
                            self.mask_paths.append(mask_path if os.path.exists(mask_path) else None)
                            # self.masks.append(Image.open(mask_path).convert("L") if os.path.exists(mask_path) else None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # image = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        original_img_size = image.size
        mask = None
        if self.transform_x:
            image = self.transform_x(image)
            
        # 마스크 로드 (None일 경우 이미지와 동일한 크기의 0으로 채워진 텐서 생성)
        if self.mask_paths[idx] is not None:
            # mask = self.masks[idx]
            mask = Image.open(self.mask_paths[idx])
            if self.transform_mask:
                mask = self.transform_mask(mask)
        else:
            # mask가 None일 경우 0으로 채워진 텐서 생성 (이미지와 동일한 크기)
            _, width, height = image.shape  # PIL 이미지의 크기 가져오기
            mask = torch.zeros((1, height, width), dtype=torch.float32)  # (C, H, W) 형식 유지

        return image, mask, label, original_img_size, img_path