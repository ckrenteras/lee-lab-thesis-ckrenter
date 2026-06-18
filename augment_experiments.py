import copy
import csv
import os

import cv2
import torch
import torch.utils.data as data
import torch.optim as optim
import segmentation_models_pytorch as smp
import argparse
from torch.cuda.amp import GradScaler

import datasets
import metrics
import augmentations as T

# ======================================================================
# Device
# ======================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================================================================
# Dataset (fixed split wont never changes across experiments)
# ======================================================================
OCTA3MM_TRAIN_SIZE = 140
OCTA3MM_TEST_SIZE  = 50
OCTA3MM_VAL_SIZE   = 10

TEST_SET      = datasets.OCTA3MM_Dataset(split='test')
TRAIN_VAL_SET = datasets.OCTA3MM_Dataset(split='train')
TRAIN_SET, VAL_SET = data.random_split(
    TRAIN_VAL_SET, [OCTA3MM_TRAIN_SIZE, OCTA3MM_VAL_SIZE],
    generator=torch.Generator().manual_seed(0)
)
TEST_LOADER = data.DataLoader(TEST_SET, batch_size=OCTA3MM_TEST_SIZE, shuffle=False, num_workers=3, pin_memory=True, persistent_workers=True)
VAL_LOADER  = data.DataLoader(VAL_SET,  batch_size=OCTA3MM_VAL_SIZE,  shuffle=False, num_workers=3, pin_memory=True, persistent_workers=True)

RESULTS_DIR = os.path.join('results', 'aug_exp')

# ======================================================================
# csv field names
# ======================================================================
EPOCH_FIELDNAMES   = ['epoch', 'train_loss', 'val_loss',
                       'val_dice', 'val_jaccard', 'val_bacc', 'val_auc']
SUMMARY_FIELDNAMES = ['model',
                       'test_loss', 'val_loss',
                       'test_dice', 'val_dice',
                       'test_jaccard', 'val_jaccard',
                       'test_bacc', 'val_bacc',
                       'test_auc', 'val_auc', 'epoch']

# ======================================================================
# train consts
# ======================================================================
NUM_EPOCHS = 200
PATIENCE   = 50
SEEDS      = list(range(5))

# ======================================================================
# Optimal hyperparameters use the ones from the optuna study results
# before running augmentation exps
# ======================================================================
OPTIMAL_PARAMS = {
    'unet': dict(
        encoder_name='resnet18',
        encoder_depth=4,
        encoder_weights='imagenet',
        in_channels=1,
        classes=1,
        decoder_channels=(256, 128, 64, 32, 16)[:4],
        batch_size=8,
        lr=1e-3,
    ),
    'manet': dict(
        encoder_name='resnet50',
        encoder_depth=3,
        encoder_weights='imagenet',
        in_channels=1,
        classes=1,
        decoder_channels=(256, 128, 64, 32, 16)[:3],
        batch_size=8,
        lr=1e-3,
    ),
}

# ======================================================================
# Augmentation experiment definitions
# Each entry: tag (unique str), augmentation callable or None,
#             augment_label (bool), double_clean (bool).
#
# double_clean=True: concatenate TRAIN_SET with itself (same data, 2×)
# augmentation!=None: concatenate TRAIN_SET with augmented copy
# ======================================================================

BASELINES = [
    {'tag': 'baseline_single', 'augmentation': None, 'augment_label': False, 'double_clean': False},
    {'tag': 'baseline_double', 'augmentation': None, 'augment_label': False, 'double_clean': True},
]

ROTATIONS = [
    {'tag': 'rotate_90_cw',  'augmentation': T.Rotation(cv2.ROTATE_90_CLOCKWISE),       'augment_label': True, 'double_clean': False},
    {'tag': 'rotate_180',    'augmentation': T.Rotation(cv2.ROTATE_180),                 'augment_label': True, 'double_clean': False},
    {'tag': 'rotate_90_ccw', 'augmentation': T.Rotation(cv2.ROTATE_90_COUNTERCLOCKWISE), 'augment_label': True, 'double_clean': False},
]

FLIPS = [
    {'tag': 'flip_horizontal', 'augmentation': T.ImFlip(flip_code=1), 'augment_label': True, 'double_clean': False},
    {'tag': 'flip_vertical',   'augmentation': T.ImFlip(flip_code=0), 'augment_label': True, 'double_clean': False},
]

GAUSSES = [
    {'tag': 'gauss_003', 'augmentation': T.GaussianNoise(std_dev=0.003), 'augment_label': False, 'double_clean': False},
    {'tag': 'gauss_005', 'augmentation': T.GaussianNoise(std_dev=0.005), 'augment_label': False, 'double_clean': False},
    {'tag': 'gauss_01',  'augmentation': T.GaussianNoise(std_dev=0.01),  'augment_label': False, 'double_clean': False},
    {'tag': 'gauss_02',  'augmentation': T.GaussianNoise(std_dev=0.02),  'augment_label': False, 'double_clean': False},
    {'tag': 'gauss_05',  'augmentation': T.GaussianNoise(std_dev=0.05),  'augment_label': False, 'double_clean': False},
]

CONTRAST_ALPHAS = [0.8, 0.9, 1.0, 1.1, 1.2]
CONTRAST_BETAS  = [-20, -10, 0, 10, 20]
DEFORM_SIGMAS   = [2, 3, 4, 5, 6]
DEFORM_POINTS   = [3, 4, 5, 6]


def make_contrast_args(alphas, betas):
    contrasts = []
    for alpha in alphas:
        for beta in betas:
            contrasts.append({
                'tag':           f'contrast_a{alpha}_b{beta}',
                'augmentation':  T.ContrastBrightness(alpha=alpha, beta=beta),
                'augment_label': False,
                'double_clean':  False,
            })
    return contrasts


def make_deform_args(sigmas, points):
    deforms = []
    for sigma in sigmas:
        for point in points:
            deforms.append({
                'tag':           f'deform_s{sigma}_p{point}',
                'augmentation':  T.ElasticDeform(sigma=sigma, points=point),
                'augment_label': True,
                'double_clean':  False,
            })
    return deforms


# ======================================================================
# Training helpers
# ======================================================================

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler):
    model.train()
    total_loss, n = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), targets.squeeze(1).float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n += 1
    scheduler.step()
    return total_loss / n


def eval_epoch(model, loader, criterion):
    model.eval()
    totals = dict(loss=0.0, dice=0.0, jaccard=0.0, balanced_acc=0.0, auc=0.0)
    n = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(1), targets.squeeze(1).float())
            totals['loss'] += loss.item()
            probs = torch.sigmoid(outputs.float()).squeeze(1)
            preds = (probs > 0.5).long()
            tgt = targets.squeeze(1).long()
            totals['dice'] += metrics.dice(preds, tgt)
            totals['jaccard'] += metrics.jaccard(preds, tgt).item()
            totals['balanced_acc'] += metrics.bacc(preds, tgt).item()
            totals['auc'] += metrics.auc(probs, tgt).item()
            n += 1
    return {k: v / n for k, v in totals.items()}


def build_train_loader(seed, augmentation=None, augment_label=False,
                       double_clean=False, batch_size=8):
    if double_clean:
        combined = data.ConcatDataset([TRAIN_SET, TRAIN_SET])
    elif augmentation is None:
        combined = TRAIN_SET
    else:
        # Create augmented copy of the same 140-image train split
        aug_train_val = datasets.OCTA3MM_Dataset(
            split='train', augmentation=augmentation, augment_label=augment_label
        )
        aug_train, _ = data.random_split(
            aug_train_val, [OCTA3MM_TRAIN_SIZE, OCTA3MM_VAL_SIZE],
            generator=torch.Generator().manual_seed(0)
        )
        combined = data.ConcatDataset([TRAIN_SET, aug_train])

    return data.DataLoader(
        combined, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=3, pin_memory=True, persistent_workers=True,
    )


def make_model(arch, params):
    net_params = {k: v for k, v in params.items() if k not in ('batch_size', 'lr')}
    if arch == 'unet':
        return torch.compile(smp.Unet(**net_params).to(device))
    if arch == 'manet':
        return torch.compile(smp.MAnet(**net_params).to(device))
    raise ValueError(f"Unknown arch: {arch}")


def already_done(summary_path, net_name):
    """Returns True if net_name already has a row in the summary CSV."""
    if not os.path.isfile(summary_path):
        return False
    with open(summary_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('model') == net_name:
                return True
    return False


def train_and_evaluate(arch, tag, seed, train_loader, summary_writer, summary_file):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    net_name = f'{arch}_{tag}_s{seed}'
    params   = OPTIMAL_PARAMS[arch]
    model    = make_model(arch, params)
    criterion = metrics.DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = GradScaler()

    per_epoch_dir = os.path.join(RESULTS_DIR, arch, 'per-epoch')
    os.makedirs(per_epoch_dir, exist_ok=True)
    epoch_csv = os.path.join(per_epoch_dir, f'{net_name}_epochs.csv')

    best_val_dice    = -1.0
    best_row         = None
    best_state_dict  = None
    patience_counter = 0

    with open(epoch_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_FIELDNAMES)
        writer.writeheader()

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss  = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler)
            val_metrics = eval_epoch(model, VAL_LOADER, criterion)

            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_dice': val_metrics['dice'],
                'val_jaccard': val_metrics['jaccard'],
                'val_bacc': val_metrics['balanced_acc'],
                'val_auc':val_metrics['auc'],
            }
            writer.writerow(row)
            f.flush()

            if epoch % 10 == 0:
                print(f"  [{net_name}]  epoch={epoch:3d}  "
                      f"train_loss={train_loss:.4f}  "
                      f"val_loss={val_metrics['loss']:.4f}  "
                      f"val_dice={val_metrics['dice']:.4f}")

            if val_metrics['dice'] > best_val_dice:
                best_val_dice    = val_metrics['dice']
                best_row         = row
                best_state_dict  = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  [{net_name}]  Early stopping at epoch {epoch}")
                    break

    # reload best weights from memory and evaluate on held-out test set
    model.load_state_dict(best_state_dict)
    test_metrics = eval_epoch(model, TEST_LOADER, criterion)

    summary_row = {
        'model': net_name,
        'test_loss': test_metrics['loss'],
        'val_loss': best_row['val_loss'],
        'test_dice': test_metrics['dice'],
        'val_dice': best_row['val_dice'],
        'test_jaccard': test_metrics['jaccard'],
        'val_jaccard': best_row['val_jaccard'],
        'test_bacc': test_metrics['balanced_acc'],
        'val_bacc': best_row['val_bacc'],
        'test_auc': test_metrics['auc'],
        'val_auc': best_row['val_auc'],
        'epoch': best_row['epoch'],
    }
    summary_writer.writerow(summary_row)
    summary_file.flush()

    print(f"\n[{arch.upper()}] {net_name} COMPLETE  "
          f"best_epoch={best_row['epoch']}  "
          f"val_dice={best_row['val_dice']:.4f}  "
          f"test_dice={test_metrics['dice']:.4f}\n")

    del model
    torch.cuda.empty_cache()


def run_arch(arch, all_augs):
    summary_dir  = os.path.join(RESULTS_DIR, arch, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f'{arch}_aug_summary.csv')

    file_exists = os.path.isfile(summary_path)
    with open(summary_path, 'a', newline='') as summary_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDNAMES)
        if not file_exists:
            summary_writer.writeheader()

        for aug in all_augs:
            for seed in SEEDS:
                net_name = f'{arch}_{aug["tag"]}_s{seed}'
                if already_done(summary_path, net_name):
                    print(f"  [{net_name}] already complete, skipping.")
                    continue

                print(f"\n{'='*65}")
                print(f"[{arch.upper()}]  aug={aug['tag']}  seed={seed}")
                print('='*65)

                train_loader = build_train_loader(
                    seed,
                    augmentation=aug['augmentation'],
                    augment_label=aug['augment_label'],
                    double_clean=aug['double_clean'],
                    batch_size=OPTIMAL_PARAMS[arch]['batch_size'],
                )
                train_and_evaluate(
                    arch, tag=aug['tag'], seed=seed,
                    train_loader=train_loader,
                    summary_writer=summary_writer,
                    summary_file=summary_file,
                )

    print(f"\n[{arch.upper()}] All experiments done. Summary: {summary_path}")


# ======================================================================
# Entry point
# ======================================================================
if __name__ == '__main__':
    contrasts = make_contrast_args(CONTRAST_ALPHAS, CONTRAST_BETAS)
    deforms   = make_deform_args(DEFORM_SIGMAS, DEFORM_POINTS)
    all_augs  = BASELINES + ROTATIONS + FLIPS + GAUSSES + contrasts + deforms

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=['unet', 'manet'], required=True,
                        help='Architecture to run augmentation experiments for.')
    args = parser.parse_args()

    run_arch(args.arch, all_augs)
    print("Done!")
