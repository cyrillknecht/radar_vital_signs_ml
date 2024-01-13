"""
Run this to do a complete preprocessing, training, and inference run.
"""
import wandb
import hydra

from omegaconf import DictConfig
from pipeline.preprocessing import preprocess
from pipeline.training import training
from pipeline.inference import inference
from pipeline.testing import testing


def run_training_pipeline(cfg,
                          train_subjects=None,
                          val_subjects=None,
                          test_subjects=None,
                          left_out_subject=None):
    """
    Run the complete pipeline.
    """

    if train_subjects is None:
        train_subjects = [0]
    if val_subjects is None:
        val_subjects = [22]
    if test_subjects is None:
        test_subjects = [23, 24]

    preprocess(target_dir=cfg.dirs.data_dir,
               train_subjects=train_subjects,
               val_subjects=val_subjects,
               test_subjects=test_subjects,
               multi_dim=cfg.preprocessing.multi_dim,
               mode=cfg.preprocessing.mode,
               data_dir=cfg.dirs.unprocessed_data_dir)

    run_name = training(cfg, left_out_subject=left_out_subject)

    return run_name


def run_test_pipeline(cfg, model_path=None):
    if model_path:
        cfg.inference.model_path = "outputs/" + model_path + ".ckpt"
    if cfg.testing.wandb_log:
        wandb.login(key=cfg.wandb.api_key)
        wandb.init(project=cfg.wandb.project_name, name=model_path)
    inference(cfg)
    testing(cfg.dirs.data_dir, cfg.testing.plot, cfg.testing.prominence, cfg.testing.wandb_log)


def full_pipeline(cfg,
                  train_subjects=None,
                  val_subjects=None,
                  test_subjects=None,
                  left_out_subject=None):
    run_name = run_training_pipeline(cfg=cfg,
                                     train_subjects=train_subjects,
                                     val_subjects=val_subjects,
                                     test_subjects=test_subjects,
                                     left_out_subject=left_out_subject)
    run_test_pipeline(cfg, run_name)


def leave_one_out_training(cfg):
    """
    Run the complete pipeline for all subjects.
    """
    for i in range(1, 24):
        train_subjects = [x for x in range(25) if x != i]
        val_subjects = [0]
        test_subjects = [i]
        print("Now running leave-one-out for subject: ", i)
        full_pipeline(cfg,
                      train_subjects=train_subjects,
                      val_subjects=val_subjects,
                      test_subjects=test_subjects,
                      left_out_subject=i)


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    hydra.output_subdir = None  # Prevent hydra from creating a new folder for each run
    # Run only one training run
    # run_training_pipeline(cfg)

    # Run only one test run
    # run_test_pipeline(cfg)

    # Run complete pipeline
    # full_pipeline(cfg)

    # Run complete training with all subjects as test subjects
    leave_one_out_training(cfg)


if __name__ == "__main__":
    main()
