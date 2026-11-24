import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import argparse
import os
import yaml
from tqdm import tqdm
import time

# Custom Imports
from src.models.siamese_main import AttentionalSiameseNetwork
from src.data.lfw_dataset import LFWDataset, PairListDataset
from src.data.splits import read_pairs, read_people
from src.data.transforms import get_transforms
from src.utils.metrics import accuracy_at, calculate_accuracy, predict_probs, roc_auc
from src.utils.seed import set_seed
from src.utils.visualization import save_attention_map, plot_loss_curve

# --- Load Configuration ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Hyperparameters
BATCH_SIZE = config.get("batch_size", 32)
LEARNING_RATE = config.get("learning_rate", 1e-4)
NUM_EPOCHS = config.get("num_epochs", 20)
WEIGHT_DECAY = config.get("weight_decay", 1e-5)
DATASET_PATH = config.get("dataset_path", "data/lfw")
CHECKPOINT_DIR = config.get("checkpoint_dir", "checkpoints")
LOG_DIR = config.get("log_dir", "logs")
DROPOUT_PROB = config.get("dropout_prob", 0.0)
INPUT_SHAPE = tuple(config.get("input_shape", [224, 224]))
UNFREEZE_EPOCH = config.get("unfreeze_epoch", 5)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model):
    """The ONE place the optimizer and LR scheduler are built.

    Used both at the start and when the backbone is unfrozen (the optimizer must
    be rebuilt so it includes the newly trainable parameters). Two hand-copied
    constructions drifted before: only the unfreeze copy passed `verbose=True`,
    which newer PyTorch rejects, so training crashed at UNFREEZE_EPOCH.
    """
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    return optimizer, scheduler

def train_one_epoch(model, loader, criterion, optimizer, epoch, writer, max_batches=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    # scaler = torch.amp.GradScaler('cuda') # AMP Disabled due to NaN instability


    loop = tqdm(loader, desc=f"Train Epoch [{epoch+1}/{NUM_EPOCHS}]")

    for batch_idx, (img1, img2, labels) in enumerate(loop):
        if max_batches is not None and batch_idx >= max_batches:
            break  # smoke-test mode (CPU)
        n_batches += 1
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        labels = labels.to(DEVICE).float().view(-1, 1)
        
        
        optimizer.zero_grad()
        
        # Forward (No AMP)
        outputs, attn_map = model(img1, img2)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        
        # Gradient Clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        # scaler.update()
        
        # Metrics
        # Apply sigmoid to logits for accuracy calculation
        probs = torch.sigmoid(outputs)
        acc = calculate_accuracy(probs, labels)
        running_loss += loss.item()
        running_acc += acc.item()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item(), acc=acc.item())
        
        # Logging
        global_step = epoch * len(loader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
            writer.add_scalar('Training/Batch_Accuracy', acc.item(), global_step)

    avg_loss = running_loss / n_batches
    avg_acc = running_acc / n_batches
    return avg_loss, avg_acc

def validate(model, val_loader, val_dataset, criterion, epoch):
    """Validation on the FIXED val pairs via the shared eval helper.
    Returns (loss, auc, acc@0.5). The test split is never touched during training."""
    probs, labels, val_loss = predict_probs(model, val_loader, DEVICE, criterion=criterion)
    val_auc = roc_auc(labels, probs)
    val_acc = accuracy_at(labels, probs, 0.5)

    # Attention visualisation for the first (fixed) val pair
    img1, img2, label = val_dataset[0]
    with torch.inference_mode():
        _, attn_map = model(img1.unsqueeze(0).to(DEVICE), img2.unsqueeze(0).to(DEVICE))
    save_attention_map(img1, img2, attn_map[0], float(probs[0]), label.item(),
                       epoch + 1, 0, save_dir=os.path.join(LOG_DIR, "vis"))
    return val_loss, val_auc, val_acc

def main(max_batches=None, num_workers=8):
    set_seed(config["seed"])
    # Ensure directories exist (after CLI overrides are applied)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "vis"), exist_ok=True)
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cpu':
        print("WARNING: You are training on CPU. This will be slow!")
        print("Install CUDA or use a machine with Nvidia GPU for faster results.")
    else:
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    
    # 1. Data Preparation
    print("Initializing Datasets...")
    splits_dir = config["splits_dir"]
    # Training: random on-the-fly pairs over TRAIN people only
    train_dataset = LFWDataset(
        root_dir=DATASET_PATH,
        transform=get_transforms(split='train', input_size=INPUT_SHAPE),
        people=read_people(os.path.join(splits_dir, "train_people.txt")),
        cache_images=False
    )

    # Validation: FIXED pairs over VAL people (disjoint from train and test).
    # Used for checkpoint selection and LR scheduling. Test is only read by evaluate.py.
    val_dataset = PairListDataset(
        DATASET_PATH,
        read_pairs(os.path.join(splits_dir, "val_pairs.csv")),
        transform=get_transforms(split='test', input_size=INPUT_SHAPE),
    )

    loader_kw = dict(num_workers=num_workers, pin_memory=DEVICE.type == "cuda")
    if num_workers > 0:
        loader_kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kw)
    
    # 2. Model, Loss, Optimizer
    # We freeze the backbone to speed up training and prevent overfitting on small data
    model = AttentionalSiameseNetwork(input_shape=INPUT_SHAPE, dropout_prob=DROPOUT_PROB, freeze_backbone=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer, scheduler = build_optimizer(model)
    
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # Store metrics for plotting later
    train_losses = []
    val_losses = []
    
    best_auc = -1.0

    # 3. Training Loop
    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        
        # Unfreeze Backbone Logic
        if epoch == UNFREEZE_EPOCH:
            print(f"\n[INFO] Reached epoch {epoch}. Unfreezing backbone for fine-tuning!")
            model.unfreeze_backbone()
            # We must re-initialize the optimizer to include the newly unfrozen parameters
            # We keep the low learning rate (or make it even lower if desired)
            optimizer, scheduler = build_optimizer(model)

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, max_batches)
        train_losses.append(train_loss)

        # Validate every epoch on the fixed val pairs
        val_loss, val_auc, val_acc = validate(model, val_loader, val_dataset, criterion, epoch)
        val_losses.append(val_loss)

        # Step Scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}"
              f"\tTrain Loss: {train_loss:.4f} \tTrain Acc: {train_acc:.4f}"
              f"\tVal Loss: {val_loss:.4f} \tVal AUC: {val_auc:.4f} \tVal Acc@0.5: {val_acc:.4f}")

        # TensorBoard
        writer.add_scalars('Epoch/Loss', {'Train': train_loss, 'Val': val_loss}, epoch+1)
        writer.add_scalars('Epoch/Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch+1)
        writer.add_scalar('Epoch/Val_AUC', val_auc, epoch+1)

        # Checkpoint selection on val AUC (threshold-free); test is never consulted
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
            print(f"Saved Best Model (val AUC {val_auc:.4f})")

        # Save regular checkpoint
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/last_model.pth")

    print("Training Complete!")
    writer.close()
    
    # Plot final loss curve
    plot_loss_curve(train_losses, val_losses, save_path=os.path.join(LOG_DIR, "loss_curve.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="override num_epochs from config.yaml")
    ap.add_argument("--max-batches", type=int, default=None, help="cap train batches per epoch (CPU smoke test)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--checkpoint-dir", default=None, help="override checkpoint_dir (keep smoke runs out of checkpoints/)")
    ap.add_argument("--log-dir", default=None, help="override log_dir")
    args = ap.parse_args()
    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.checkpoint_dir is not None:
        CHECKPOINT_DIR = args.checkpoint_dir
    if args.log_dir is not None:
        LOG_DIR = args.log_dir
    main(max_batches=args.max_batches, num_workers=args.num_workers)