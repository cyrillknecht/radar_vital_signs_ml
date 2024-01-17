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
from lightning.pytorch import seed_everything


def run_training_pipeline(cfg,
                          train_subjects=None,
                          val_subjects=None,
                          test_subjects=None,
                          left_out_subject=None):
    """
    Run the complete pipeline.
    """

    if not train_subjects or not val_subjects or not test_subjects:
        print("No subjects provided. Using default values.")

    if train_subjects is None:
        train_subjects = [x for x in range(25) if x != 0 and x != 1]
        print("No train subjects provided. Using default values: ", train_subjects)
    if val_subjects is None:
        val_subjects = [0]
        print("No val subjects provided. Using default values: ", val_subjects)
    if test_subjects is None:
        test_subjects = [1]
        print("No test subjects provided. Using default values: ", test_subjects)

    preprocess(target_dir=cfg.dirs.data_dir,
               train_subjects=train_subjects,
               val_subjects=val_subjects,
               test_subjects=test_subjects,
               multi_dim=cfg.preprocessing.multi_dim,
               mode=cfg.preprocessing.mode,
               data_dir=cfg.dirs.unprocessed_data_dir,
               slice_duration=cfg.preprocessing.slice_duration)

    run_name = training(cfg, left_out_subject=left_out_subject)

    return run_name


def run_test_pipeline(cfg, model_path=None):
    if model_path:
        cfg.inference.model_path = "outputs/" + model_path + ".ckpt"
    if cfg.testing.wandb_log:
        wandb.login(key=cfg.wandb.api_key)
        wandb.init(project=cfg.wandb.project_name, name=model_path)
    print("Testing model: ", cfg.inference.model_path)
    inference(cfg)
    results = testing(cfg.dirs.data_dir, cfg.testing.plot, cfg.testing.prominence, cfg.testing.wandb_log)

    cfg.inference.model_path = None  # Reset model path to prevent it from being used in the next run

    return results


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
    results = run_test_pipeline(cfg, run_name)

    return results


def leave_one_out_training(cfg):
    """
    Run the complete pipeline for all subjects.
    """
    results = []
    for i in range(1, 24):
        train_subjects = [x for x in range(1, 25) if x != i]
        val_subjects = [0]
        test_subjects = [i]
        print("Now running leave-one-out for subject: ", i)
        result = full_pipeline(cfg,
                               train_subjects=train_subjects,
                               val_subjects=val_subjects,
                               test_subjects=test_subjects,
                               left_out_subject=i)
        results.append(result)

    # Average the results list of dicts into one dict
    avg_results = {}
    for key in results[0].keys():
        avg_results[key] = sum(d[key] for d in results) / len(results)
    print(avg_results)

    return avg_results


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(42, workers=True)
    hydra.output_subdir = None  # Prevent hydra from creating a new folder for each run

    # Error checking
    model_type = cfg.model
    if cfg.models[model_type].input_size != 1 and not cfg.preprocessing.multi_dim:
        raise ValueError("Input size must be 1 if multi_dim is False.")
    if cfg.models[model_type].output_size != 1 and not cfg.preprocessing.mode == "binary classification":
        raise ValueError("Output size must be 1 if mode is not binary classification.")

    # Run only one training run
    if cfg.main.mode == "train":
        print("Running training pipeline...")
        run_training_pipeline(cfg)

    # Run only one test run
    elif cfg.main.mode == "test":
        print("Running test pipeline...")
        run_test_pipeline(cfg)

    # Run the complete pipeline
    elif cfg.main.mode == "full":
        print("Running full pipeline...")
        full_pipeline(cfg)

    # Run complete training with all subjects as test subjects
    elif cfg.main.mode == "leave_one_out":
        print("Running leave-one-out training...")
        leave_one_out_training(cfg)

    else:
        raise ValueError("Invalid mode. Must be one of: train, test, full, leave_one_out")


if __name__ == "__main__":
    main()
