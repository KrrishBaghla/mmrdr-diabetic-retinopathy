import os
import time
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
# 1. GRAPH NEURAL NETWORK COMPONENTS
# ==========================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        # x: [batch_size, num_nodes, in_features]
        # adj: [batch_size, num_nodes, num_nodes]
        support = torch.matmul(x, self.weight) 
        output = torch.matmul(adj, support)   
        return output + self.bias

class ResNetGCNClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNetGCNClassifier, self).__init__()
        # Use ResNet-18 to keep memory usage low on 6 GB GPU.
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: [B, 512, 8, 8] for 256x256 input

        # GCN layers to process relational features
        self.gcn1 = GraphConvolution(in_features=512, out_features=256)
        self.gcn2 = GraphConvolution(in_features=256, out_features=128)
        self.residual_proj = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.2)
        
        # Classification Head
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.backbone(x) # [Batch, 2048, 16, 16]
        batch_size, C, H, W = features.size()
        
        # Flatten spatial map to graph nodes (16x16 = 256 nodes)
        nodes = features.view(batch_size, C, H * W).permute(0, 2, 1) # [Batch, 256, 2048]
        
        # Compute self-attention-based dynamic Adjacency Matrix
        nodes_norm = F.normalize(nodes, p=2, dim=-1)
        adj = torch.bmm(nodes_norm, nodes_norm.permute(0, 2, 1)) # [Batch, 256, 256]
        adj = adj + torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        adj = F.softmax(adj, dim=-1)
        
        # Reason over features via Graph Convolutions
        x_graph = F.relu(self.gcn1(nodes, adj))   # [Batch, 256, 512]
        x_graph = self.dropout(x_graph)
        x_graph = F.relu(self.gcn2(x_graph, adj))  # [Batch, 256, 128]
        x_graph = self.dropout(x_graph)

        # Residual path to preserve local structure and reduce over-smoothing
        residual = self.residual_proj(nodes)
        x_graph = x_graph + 0.1 * residual
        
        # Global Pooling (Average node features)
        out = x_graph.mean(dim=1) # [Batch, 128]
        return self.fc(out)

# ==========================================
# 2. DATASET MANAGEMENT
# ==========================================
class MMRDRDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['image']))
        image = Image.open(img_path).convert('RGB')
        label = int(row['grade'])
        
        if self.transform:
            image = self.transform(image)
        return image, label

def resolve_data_dir(base_path):
    """Resolve dataset paths relative to this script, not the current shell CWD."""
    if os.path.isabs(base_path):
        return base_path
    return os.path.normpath(os.path.join(os.path.dirname(__file__), base_path))


def execute_mmrdr_split(base_path, modality):
    data_dir = resolve_data_dir(base_path)
    modality_label = 'FP' if modality.lower() == 'cfp' else modality.upper()
    file_path = os.path.join(data_dir, f'MMRDR-{modality.upper()}', f'{modality_label}.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical Error: Metadata sheet not found at {file_path}")
        
    df = pd.read_csv(file_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Production Train Script for ResNet-GCN Hybrid")
    parser.add_argument('--data_dir', type=str, default='MMRDR', help='Root dataset folder path')
    parser.add_argument('--modality', type=str, default='cfp', choices=['cfp'], help='Data modality (CFP only for this run)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for 6 GB GPU memory')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size for memory-safe training')
    parser.add_argument('--epochs', type=int, default=25, help='Maximum training epochs before early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Stop if validation accuracy does not improve for this many epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='CPU worker cores for data loading parallelism')
    parser.add_argument('--output_model', type=str, default='best_resnet_gcn_cfp.pth', help='Path destination for the best checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == 'cuda'
    print(f"Execution Target Device: {device} | AMP Enabled: {use_amp}")

    # Prepare splits
    data_dir = resolve_data_dir(args.data_dir)
    train_df, test_df = execute_mmrdr_split(data_dir, args.modality)
    img_dir = os.path.join(data_dir, f'MMRDR-{args.modality.upper()}')

    # Dynamic diagnostic transformation pipeline
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Cluster-ready high throughput loaders
    train_loader = DataLoader(
        MMRDRDataset(train_df, img_dir, train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        MMRDRDataset(test_df, img_dir, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model, Optimizer, and AMP Scaler
    model = ResNetGCNClassifier(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_acc = 0.0
    epochs_without_improve = 0
    best_state = None

    print(f"Starting CFP training for up to {args.epochs} epochs with patience={args.patience}...")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs} [train]")
        for i, (images, labels) in train_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation Phase
        model.eval()
        correct, total = 0, 0
        val_bar = tqdm(test_loader, total=len(test_loader), desc=f"Epoch {epoch + 1}/{args.epochs} [val]")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_bar.set_postfix(accuracy=f"{correct / total:.4f}" if total else "0.0000")

        epoch_acc = correct / total
        avg_loss = running_loss / len(train_loader)
        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch + 1}/{args.epochs}] -> Loss: {avg_loss:.4f} | Validation Acc: {epoch_acc:.4f} | Timeline: {epoch_elapsed:.1f}s")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, args.output_model)
            epochs_without_improve = 0
            print(f"==> Auto-saved best CFP model to {args.output_model} with Acc: {best_acc:.4f}")
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs without validation improvement.")
            break

    print(f"Training Complete! Best CFP validation accuracy: {best_acc:.4f} | Saved model: {args.output_model}")

if __name__ == '__main__':
    main()