

Run param search:
mkdir -p logs
sbatch --export=ARCH=unet submit_optuna_hyperparams.sh
sbatch --export=ARCH=manet submit_optuna_hyperparams.sh

after merge: 
head -1 results/hyperparam/unet/unet_hyperparam_0.csv > unet_hyperparam.csv
tail -n +2 -q results/hyperparam/unet/unet_hyperparam_*.csv >> unet_hyperparam.csv

head -1 results/hyperparam/manet/manet_hyperparam_0.csv > manet_hyperparam.csv
tail -n +2 -q results/hyperparam/manet/manet_hyperparam_*.csv >> manet_hyperparam.csv


Run depth-held-fixed search (e.g. depth=5, only varying encoder/batch/lr):
sbatch --array=0-1 --export=ARCH=unet,DEPTH=5,TAG=depth5,NTRIALS=180 submit_optuna_hyperparams.sh

after merge (same commands as before — the glob already picks up tagged files,
and depth is recoverable from the `model` column, e.g. unet_depth5_s3_resnet18_d5_b8_lr0):
head -1 results/hyperparam/unet/unet_hyperparam_0.csv > unet_hyperparam.csv
tail -n +2 -q results/hyperparam/unet/unet_hyperparam_*.csv >> unet_hyperparam.csv


RUN AUG EXPS:
sbatch submit_aug_experiments.sh
