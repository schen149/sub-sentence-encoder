import glob
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from arguments import create_argparser, make_experiment_id
from model.pl_subencoder import LitSubEncoder
from dataset.prop_pairs import PropPairDataset


def main(args):
    # weirdness with HuggingFace tokenizer when processing things in parallel
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Set seed for each worker
    pl.seed_everything(args.random_seed, workers=True)

    # create experiment_dir and load model
    if args.experiment_id is None:
        args.experiment_id = make_experiment_id(args)

    experiment_dir = os.path.join(args.output_dir, args.experiment_id)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    model = LitSubEncoder(args)
    dm = PropPairDataset(
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        val_data_path=args.val_data_path,
        model_name_or_path=args.model_name,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.val_batch_size,
        max_seq_length=args.max_seq_length
    )

    # Wandb logger
    logger = WandbLogger(
        project=args.project_name,
        name=f"{args.experiment_id}",
        save_dir=experiment_dir,
    )

    logger.watch(model)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # compute validation if needed, otherwise just skip it and save
    # every `period` checkpoints
    if args.validate:
        limit_val_batches = 1.0
        checkpoint_callback = ModelCheckpoint(
            dirpath=experiment_dir,
            monitor="val_loss",
            save_top_k=args.save_top_k_ckpts,
            mode="min",
            filename=os.path.join(
                args.project_name, "epoch={epoch}-step={step}-val_loss={val_loss:.2f}"
            ),
        )
    else:
        limit_val_batches = 0.0
        checkpoint_callback = ModelCheckpoint(
            dirpath=experiment_dir,
            monitor=None,
            save_top_k=-1,
            every_n_epochs=args.period
        )

    precision = int(args.precision) if args.precision != "bf16" else "bf16"
    trainer = pl.Trainer(
        default_root_dir=experiment_dir,
        max_epochs=args.num_epoch,
        logger=logger,
        enable_checkpointing=True,
        gpus=args.gpus,
        strategy='ddp' if args.gpus > 1 else None,
        precision=precision,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=args.validate_every if args.validate else 1,
        callbacks=[lr_monitor, checkpoint_callback]
    )

    if args.train:
        trainer.fit(model, datamodule=dm)

    if args.evaluate:
        trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    args = create_argparser()
    main(args)
