import os
import ast
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# 1. THE LOSS PARADIGM: CLASS-BALANCED FOCAL LOSS
# ==========================================
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(ClassBalancedFocalLoss, self).__init__()
        self.alpha = alpha  # Expected to be a tensor of normalized weights per class
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==========================================
# 2. THE CUSTOM ARCHITECTURE: MULTI-SCALE LESION-ATTENTION NET (MS-LAN)
# ==========================================
class MultiScaleLesionAttentionNet(nn.Module):
    def __init__(self, num_grades=5, num_lesions=7):
        super(MultiScaleLesionAttentionNet, self).__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        
        # Branch 1: Micro details preserved at 256x256 resolution
        self.early_features = nn.Sequential(
            resnet.conv1,    # Stride 2 downsamples 512x512 -> 256x256
            resnet.bn1,
            resnet.relu      # No maxpool here! Preserves microscopic microaneurysms
        ) # Output size: [Batch, 64, 256, 256]
        
        # Branch 2: Deep global context downsampled to 16x16 resolution
        self.deep_features = nn.Sequential(
            resnet.maxpool,  # Downsamples 256x256 -> 128x128
            resnet.layer1,   # [Batch, 256, 128, 128]
            resnet.layer2,   # [Batch, 512, 64, 64]
            resnet.layer3,   # [Batch, 1024, 32, 32]
            resnet.layer4    # [Batch, 2048, 16, 16]
        )
        
        # Multi-Scale Alignment Layers
        self.early_compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)), # Shrinks grid layout down to 16x16
            nn.Conv2d(64, 512, kernel_size=1) # Expands channels from 64 to 512
        )
        self.deep_compress = nn.Conv2d(2048, 512, kernel_size=1) # Compresses 2048 to 512
        
        # Head A: Multi-label Lesion Estimation Mask (BCE Tracking)
        self.lesion_head = nn.Sequential(
            nn.Conv2d(512, num_lesions, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Head B: DR Severity Grading Head with regularization against overfitting
        self.grade_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization barrier
            nn.Linear(256, num_grades)
        )

    def forward(self, x):
        # 1. Run the Micro detail extraction layer
        early_maps = self.early_features(x) # [B, 64, 256, 256]
        
        # 2. Complete the normal deep macro path
        deep_maps = self.deep_features(early_maps) # [B, 2048, 16, 16]
        
        # 3. Shape alignment and channel compression
        f1 = self.early_compress(early_maps) # [B, 512, 16, 16]
        f2 = self.deep_compress(deep_maps)   # [B, 512, 16, 16]
        
        # 4. Feature Fusion (Combines Micro + Macro vectors)
        fused_features = F.relu(f1 + f2) # [B, 512, 16, 16]
        
        # 5. Route features to target specific heads
        lesion_out = self.lesion_head(fused_features) # [B, num_lesions]
        grade_out = self.grade_head(fused_features)   # [B, num_grades]
        
        return grade_out, lesion_out

# ==========================================
# 3. PRODUCTION DATA PIPELINE
# ==========================================
class MMRDRMultiTaskDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = dataframe['image'].values
        self.grades = dataframe['grade'].values
        
        # Parse text lists into numpy matrix to maintain multi-worker tracking safely
        raw_lesions = [ast.literal_eval(x) for x in dataframe['lesion']]
        self.lesions = np.array(raw_lesions, dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        grade = int(self.grades[idx])
        lesion_tensor = torch.from_numpy(self.lesions[idx])
        
        if self.transform:
            image = self.transform(image)
        return image, grade, lesion_tensor

def execute_mmrdr_split(base_path, modality):
    csv_name = 'FP.csv' if modality.lower() == 'cfp' else f'{modality.upper()}.csv'
    file_path = os.path.join(base_path, f'MMRDR-{modality.upper()}', csv_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing resource path error: {file_path}")
    df = pd.read_csv(file_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

# ==========================================
# 4. TRAINING ROUTINE Execution Engine
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="MS-LAN Multi-Task Cluster Engine")
    default_data_dir = os.path.join(os.path.dirname(__file__), 'MMRDR')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Dataset root path')
    parser.add_argument('--modality', type=str, default='cfp', choices=['cfp', 'uwf', 'oct'], help='Target modality')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch sizing')
    parser.add_argument('--epochs', type=int, default=50, help='Total epoch runs')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Computation device to use')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate value')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps for low VRAM targets')
    parser.add_argument('--num_workers', type=int, default=4, help='Parallel CPU threads')
    args = parser.parse_args()

    if args.device == 'auto':
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_type = args.device

    device = torch.device('cuda:0' if device_type == 'cuda' else 'cpu')
    print(f"Ignition Target Device: {device}")
    if device_type == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    train_df, test_df = execute_mmrdr_split(args.data_dir, args.modality)
    img_dir = os.path.join(args.data_dir, f'MMRDR-{args.modality.upper()}')

    # Enhancing transforms to push optimization limits
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset loader setup optimized using system constraints
    train_loader = DataLoader(
        MMRDRMultiTaskDataset(train_df, img_dir, train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        MMRDRMultiTaskDataset(test_df, img_dir, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Calculate inverse frequency class balancing weights for Focal Loss
    counts = torch.tensor([6579, 1302, 1959, 603, 1553], dtype=torch.float32)
    class_weights = 1.0 / (counts / counts.sum())
    normalized_weights = (class_weights / class_weights.min()).to(device)

    # Initialize Core Elements
    model = MultiScaleLesionAttentionNet(num_grades=5, num_lesions=7).to(device)
    grade_criterion = ClassBalancedFocalLoss(alpha=normalized_weights, gamma=2.0)
    lesion_criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' else None

    best_acc = 0.0

    print("Executing loop...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (images, labels, lesions) in train_pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lesions = lesions.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                g_out, l_out = model(images)
                loss_g = grade_criterion(g_out, labels)
                loss_l = lesion_criterion(l_out, lesions)
                # Apply 3.0x scaling coefficient for Severity Classification dominance
                loss = (3.0 * loss_g + loss_l) / args.accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * args.accumulation_steps
            train_pbar.set_postfix(g_loss=f"{loss_g.item():.4f}", l_loss=f"{loss_l.item():.4f}")

        scheduler.step()

        # Validation Phase
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                    g_out, _ = model(images)
                    _, predicted = torch.max(g_out, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_acc = correct / total
        print(f"--> Epoch [{epoch+1}/{args.epochs}] Finished | Global Accuracy: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_ms_lan_network.pth')
            print(f"*** NEW BEST MODEL OVERWRITE RECORDED: {best_acc:.4f} ***")

    print(f"Pipeline Complete. Peak Benchmark Reached: {best_acc:.4f}")

if __name__ == '__main__':
    main()