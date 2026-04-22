import os
import torch
import timm
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import f1_score
from tqdm import tqdm

DATA_DIR = "Balanced_1500"
IMG_SIZE = 224
BATCH_SIZE = 32

QUALITY_CLASSES = ["good", "bad", "mixed"]
FRUIT_CLASSES = sorted(os.listdir(DATA_DIR))

NUM_QUALITY = len(QUALITY_CLASSES)
NUM_FRUIT = len(FRUIT_CLASSES)

EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 10

LR_STAGE1 = 5e-4
LR_STAGE2 = 5e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(35),
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.2
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5)
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

class MultiTaskDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        quality_map = {"good": 0, "bad": 1, "mixed": 2}
        fruit_map = {f: i for i, f in enumerate(FRUIT_CLASSES)}

        for fruit in FRUIT_CLASSES:
            fruit_path = os.path.join(root, fruit)

            for q, q_idx in quality_map.items():
                q_path = os.path.join(fruit_path, q)

                for img in os.listdir(q_path):
                    self.samples.append(
                        (
                            os.path.join(q_path, img),
                            fruit_map[fruit],
                            q_idx
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, fruit_lbl, quality_lbl = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, fruit_lbl, quality_lbl

dataset = MultiTaskDataset(DATA_DIR, train_tfms)

train_len = int(0.7 * len(dataset))
val_len = int(0.15 * len(dataset))
test_len = len(dataset) - train_len - val_len

train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

val_ds.dataset.transform = val_tfms
test_ds.dataset.transform = val_tfms

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

class ConvNeXtMultiTask(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            num_classes=0
        )

        feat_dim = self.backbone.num_features

        self.fruit_head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.7),
            nn.Linear(feat_dim, NUM_FRUIT)
        )

        self.quality_head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.7),
            nn.Linear(feat_dim, NUM_QUALITY)
        )

    def forward(self, x):
        feats = self.backbone(x)

        fruit_out = self.fruit_head(feats)
        quality_out = self.quality_head(feats)

        return fruit_out, quality_out

model = ConvNeXtMultiTask().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.25)

def train_one_epoch(loader, optimizer):
    model.train()
    total_loss = 0

    for imgs, f_lbl, q_lbl in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        f_lbl = f_lbl.to(device)
        q_lbl = q_lbl.to(device)

        optimizer.zero_grad()

        fo, qo = model(imgs)

        loss = criterion(fo, f_lbl) + criterion(qo, q_lbl)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(loader):
    model.eval()

    f_true, f_pred = [], []
    q_true, q_pred = [], []

    with torch.no_grad():
        for imgs, f_lbl, q_lbl in loader:
            imgs = imgs.to(device)

            fo, qo = model(imgs)

            f_pred.extend(fo.argmax(1).cpu().numpy())
            q_pred.extend(qo.argmax(1).cpu().numpy())

            f_true.extend(f_lbl.numpy())
            q_true.extend(q_lbl.numpy())

    fruit_f1 = f1_score(f_true, f_pred, average="macro")
    quality_f1 = f1_score(q_true, q_pred, average="macro")

    return fruit_f1, quality_f1

print("Stage 1 Feature Extraction")

for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = optim.Adam(
    list(model.fruit_head.parameters()) +
    list(model.quality_head.parameters()),
    lr=LR_STAGE1,
    weight_decay=1e-2
)

for epoch in range(EPOCHS_STAGE1):
    loss = train_one_epoch(train_loader, optimizer)

    f1_fruit, f1_quality = evaluate(val_loader)

    print(
        f"Epoch {epoch+1} | Loss {loss:.4f} | "
        f"FruitF1 {f1_fruit:.3f} | QualityF1 {f1_quality:.3f}"
    )

print("Stage 2 Fine Tuning")

for name, p in model.backbone.named_parameters():
    if "stages.3.blocks.2" in name:
        p.requires_grad = True
    else:
        p.requires_grad = False

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_STAGE2,
    weight_decay=1e-2
)

for epoch in range(EPOCHS_STAGE2):
    loss = train_one_epoch(train_loader, optimizer)

    f1_fruit, f1_quality = evaluate(val_loader)

    print(
        f"Epoch {epoch+1} | Loss {loss:.4f} | "
        f"FruitF1 {f1_fruit:.3f} | QualityF1 {f1_quality:.3f}"
    )

print("Final Test Evaluation")

fruit_f1, quality_f1 = evaluate(test_loader)

print(f"Test Fruit F1 : {fruit_f1:.3f}")
print(f"Test Quality F1 : {quality_f1:.3f}")

torch.save(model.state_dict(), "convnext_multitask.pth")

print("ConvNeXt multitask model saved")