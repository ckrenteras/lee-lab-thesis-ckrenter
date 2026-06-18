import optuna
import os
import csv
import shutil
import torch
import torch.utils.data as data
import torch.optim as optim
import segmentation_models_pytorch as smp

import argparse

import datasets
import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================================================================
# Define global datasets and paths here
# These will not change throughout trials (same order, same batch size, etc.)
# ======================================================================

OCTA3MM_TRAIN_SIZE = 140
OCTA3MM_TEST_SIZE = 50
OCTA3MM_VAL_SIZE = 10

TEST_SET = datasets.OCTA3MM_Dataset(split='test')
TRAIN_VAL_SET = datasets.OCTA3MM_Dataset(split='train')
TRAIN_SET, VAL_SET = data.random_split( TRAIN_VAL_SET, [OCTA3MM_TRAIN_SIZE, OCTA3MM_VAL_SIZE], 
                                       generator=torch.Generator().manual_seed(0))
TEST_LOADER = data.DataLoader(TEST_SET, batch_size=OCTA3MM_TEST_SIZE, shuffle=False)
VAL_LOADER = data.DataLoader(VAL_SET, batch_size=OCTA3MM_VAL_SIZE, shuffle=False)


RESULTS_DIR = os.path.join('results', 'hyperparam')
MODELS_DIR = os.path.join('results', 'hyperparam', 'models')

EPOCH_FIELDNAMES = ['epoch', 'train_loss', 'val_loss', 'val_dice', 'val_jaccard', 'val_bacc', 'val_auc']
BEST_FIELDNAMES  = ['model', 'test_loss', 'val_loss', 'test_dice', 'val_dice', 'test_jaccard',
                    'val_jaccard', 'test_bacc', 'val_bacc', 'test_auc', 'val_auc', 'epoch']

# ======================================================================
# Global Vars
# ======================================================================

NUM_EPOCHS = 200
PATIENCE = 50
ARCHS = ['unet', 'manet']
NUM_TRIALS = 360

HYPERPARAM_GRID = {
    'depth': [3, 4],
    'encoder_name': ['resnet18', 'resnet34', 'resnet50'],
    'batch_size': [8, 16, 32, 64],
    'start_lr': [1e-3, 5e-4, 1e-4]
}

def model_dir_for_trial(arch, tag, trial):
    """
    Reconstructs a trial's checkpoint dir (and net_name) from its arch/tag/params.
    Single source of truth so the objective (which saves checkpoints) and the
    pruning callback (which deletes them) never disagree on the path.
    """
    arch_tag = f'{arch}_{tag}' if tag else arch
    lr_idx = HYPERPARAM_GRID['start_lr'].index(trial.params['start_lr'])
    net_name = (f"{arch_tag}_s{trial.number}_{trial.params['encoder_name']}"
                f"_d{trial.params['depth']}_b{trial.params['batch_size']}_lr{lr_idx}")
    return os.path.join(MODELS_DIR, arch, net_name), net_name


def make_prune_callback(arch, tag):
    """
    Keeps disk usage bounded across a long study: after every trial, deletes
    the saved checkpoint of every COMPLETE trial except the current best, and
    deletes a trial's own checkpoint outright if the trial didn't COMPLETE
    (e.g. crashed mid-training). Re-scans from scratch each time so it's
    idempotent and safe if multiple array tasks run it concurrently.
    """
    def prune_old_checkpoints(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            try:
                model_dir, _ = model_dir_for_trial(arch, tag, trial)
                if os.path.isdir(model_dir):
                    shutil.rmtree(model_dir, ignore_errors=True)
            except (KeyError, ValueError):
                pass  # trial failed before all hyperparams were sampled
            return

        best_number = study.best_trial.number
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if t.number == best_number:
                continue
            model_dir, _ = model_dir_for_trial(arch, tag, t)
            if os.path.isdir(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
    return prune_old_checkpoints


# ======================================================================
# Helper f'ns for training
# ======================================================================

def train_epoch(model, loader, criterion, optimizer, scheduler):
    """
    trains one epoch of a specified model with specified data loader, criterion, 
    and optimizer. Returns average loss from the epoch
    """
    model.train()
    total_loss, n = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), targets.squeeze(1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    scheduler.step()
    return total_loss / n

def eval_epoch(model, loader, criterion):
    "Returns dict of averaged metrics over input loader"
    model.eval()
    totals = dict(loss=0.0, dice=0.0, jaccard=0.0, balanced_acc=0.0, auc=0.0)
    n = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), targets.squeeze(1).float())
            totals['loss'] += loss.item()

            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs > 0.5).long()
            tgt   = targets.squeeze(1).long()
            totals['dice'] += metrics.dice(preds, tgt)
            totals['jaccard'] += metrics.jaccard(preds, tgt).item()
            totals['balanced_acc'] += metrics.bacc(preds, tgt).item()
            totals['auc'] += metrics.auc(probs, tgt).item()
            n += 1
    return {k: v / n for k, v in totals.items()}


# ======================================================================
# Wrapper f'n to set hyperparams and train
# ======================================================================

def make_objective(arch, hyperparam_grid, per_epoch_dir, summary_writer, summary_file, tag=''):
    """
    Returns an Optuna objective function closed over the arch and open
    CSV writer so results are flushed incrementally after every trial.

    `tag` if set is in net_name so a namespaced run (e.g. a
    depth-held-constant sweep) wont collide with the main run
    per-epoch CSVs even though both share the
    same per-epoch/models directories.
    """
    def objective(trial):
        # trial.number runs 0–9, used as the torch seed so each trial
        # is independently reproducible
        seed = trial.number
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        depth = trial.suggest_categorical('depth', hyperparam_grid['depth'])
        encoder = trial.suggest_categorical('encoder_name', hyperparam_grid['encoder_name'])
        batch_size = trial.suggest_categorical('batch_size', hyperparam_grid['batch_size'])
        lr = trial.suggest_categorical('start_lr', hyperparam_grid['start_lr'])

        model_dir, net_name = model_dir_for_trial(arch, tag, trial)
        os.makedirs(model_dir, exist_ok=True)
        per_epoch_path = os.path.join(per_epoch_dir, f'{net_name}_epochs.csv')

        print(f"\n{'='*60}")
        print(f"[{arch.upper()}] Trial {trial.number} | seed={seed} | "
              f"{encoder}  depth={depth}  batch_size={batch_size}  lr={lr}")
        print(f"{'='*60}")

        net_params = dict(encoder_name=encoder, encoder_depth=depth,
                  encoder_weights='imagenet', in_channels=1, classes=1,
                  decoder_channels=(256, 128, 64, 32, 16)[:depth])
        model = (smp.Unet(**net_params) if arch == 'unet'
                else smp.MAnet(**net_params)).to(device)

        criterion = metrics.DiceBCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        train_loader = data.DataLoader(
            TRAIN_SET, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed)
        )

        best_val_dice    = -1.0
        best_row         = None
        patience_counter = 0

        with open(per_epoch_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=EPOCH_FIELDNAMES)
            writer.writeheader()

            for epoch in range(NUM_EPOCHS):
                train_loss  = train_epoch(model, train_loader, criterion, optimizer, scheduler)
                val_metrics = eval_epoch(model, VAL_LOADER, criterion)

                row = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_metrics['loss'],
                    'val_dice': val_metrics['dice'],
                    'val_jaccard': val_metrics['jaccard'],
                    'val_bacc': val_metrics['balanced_acc'],
                    'val_auc': val_metrics['auc'],
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
                    model.save_pretrained(model_dir)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"  [{net_name}]  Early stopping at epoch {epoch}")
                        break

        # Load the best checkpoint and evaluate on the held-out test set
        best_model = smp.from_pretrained(model_dir).to(device)
        test_metrics = eval_epoch(best_model, TEST_LOADER, criterion)

        summary_row = {
            'model':        net_name,
            'test_loss':    test_metrics['loss'],
            'val_loss':     best_row['val_loss'],
            'test_dice':    test_metrics['dice'],
            'val_dice':     best_row['val_dice'],
            'test_jaccard': test_metrics['jaccard'],
            'val_jaccard':  best_row['val_jaccard'],
            'test_bacc':    test_metrics['balanced_acc'],
            'val_bacc':     best_row['val_bacc'],
            'test_auc':     test_metrics['auc'],
            'val_auc':      best_row['val_auc'],
            'epoch':        best_row['epoch'],
        }
        summary_writer.writerow(summary_row)
        summary_file.flush()

        print(f"\n[{arch.upper()}] Trial {trial.number} COMPLETE  "
              f"best_epoch={best_row['epoch']}  "
              f"val_dice={best_row['val_dice']:.4f}  "
              f"test_dice={test_metrics['dice']:.4f}\n")

        del model, best_model
        torch.cuda.empty_cache()

        return best_val_dice

    return objective


def run_study(arch, depth=None, tag='', n_trials=None):
    per_epoch_dir = os.path.join(RESULTS_DIR, arch, 'per-epoch')
    os.makedirs(per_epoch_dir, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, arch), exist_ok=True)

    grid = dict(HYPERPARAM_GRID)
    if depth is not None:
        grid['depth'] = [depth]

    suffix = f'_{tag}' if tag else ''
    n_trials = n_trials if n_trials is not None else NUM_TRIALS
    

    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    n_workers = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', '1'))
    summary_path = os.path.join(RESULTS_DIR, arch, f'{arch}_hyperparam{suffix}_{task_id}.csv')

    print(f"\n{'#'*60}")
    print(f"  Optuna study: {arch.upper()}{suffix}  ({n_trials} trials) depth={grid['depth']}") 
    print(f"{'#'*60}\n")

    file_exists = os.path.isfile(summary_path)
    with open(summary_path, 'a', newline='') as summary_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=BEST_FIELDNAMES)
        if not file_exists:
            summary_writer.writeheader()

        sampler = optuna.samplers.TPESampler(seed=0)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                f"{RESULTS_DIR}/{arch}/optuna_study{suffix}.log"
            )
        )
        study = optuna.create_study(
            study_name=f"{arch}_study{suffix}",
            direction='maximize',
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(make_objective(arch, grid, per_epoch_dir, summary_writer, summary_file, tag),
                       n_trials=n_trials // n_workers,
                       catch=(torch.OutOfMemoryError,),
                       gc_after_trial=True,
                       callbacks=[make_prune_callback(arch, tag)])

    best = study.best_trial
    print(f"[{arch.upper()}] Study complete.")
    print(f"  Best trial : {best.number}")
    print(f"  Best val_dice: {study.best_value:.4f}")
    print(f"  Best params  : {best.params}\n")

# ======================================================================
# Main here. Run study
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=['unet', 'manet'], default=None,
                        help='Architecture to tune. Runs both if omitted.')
    parser.add_argument('--depth', type=int, default=None, help='Hold encoder_depth constant vallue for sweep')
    parser.add_argument('--tag', type=str, default='',
                        help='Namespace this run\'s study/db/CSV/model files so it '
                        'cannot collide with or mutate the default run.')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Override NUM_TRIALS for this run.')
    args = parser.parse_args()

    for arch in ([args.arch] if args.arch else ARCHS):
        run_study(arch, depth=args.depth, tag=args.tag, n_trials=args.n_trials)

    print("Done!")