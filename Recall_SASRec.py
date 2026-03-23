# Recall_SASRec.py
import os
import argparse
from logging import getLogger
from pathlib import Path

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer, ensure_dir


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sasrec_ml1m.yaml")
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--model", default="SASRec")
    parser.add_argument("--save_name", default="SASRec_ml1m_top20.pth")
    args = parser.parse_args()

    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[str(Path(args.config))],
    )

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    ensure_dir(config["checkpoint_dir"])
    save_path = os.path.join(config["checkpoint_dir"], args.save_name)
    trainer.saved_model_file = save_path

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )

    logger.info(f"best_valid_score={best_valid_score}, best_valid_result={best_valid_result}")
    logger.info(f"test_result={test_result}")
    logger.info(f"saved_model_file={trainer.saved_model_file}")


if __name__ == "__main__":
    main()
