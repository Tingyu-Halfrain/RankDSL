from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path


def export_sasrec_topk_candidates(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    k: int = 20,
    dataset_name: str | None = None,
) -> None:
    resolved_config = Path(config_path).resolve()
    project_root = resolved_config.parent.parent
    resolved_checkpoint = Path(checkpoint_path).resolve()
    resolved_output = Path(output_path).resolve()
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved_checkpoint}. Train ML-1M SASRec first or pass --checkpoint explicitly."
        )

    try:
        import numpy as np
        import torch
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import get_model
        from recbole.utils.case_study import full_sort_topk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("export_sasrec_topk_candidates requires recbole, torch, and numpy installed") from exc

    resolved_dataset_name = dataset_name or ("amazon-books" if "amazon" in resolved_config.name else "ml-1m")
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        config = Config(
            model="SASRec",
            dataset=resolved_dataset_name,
            config_file_list=[str(resolved_config)],
            config_dict={
                "data_path": str((project_root / "dataset").resolve()),
                "checkpoint_dir": str((project_root / "saved_ckpt").resolve()),
            },
        )
    finally:
        sys.argv = original_argv
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="using <built-in function len> in Series.agg cannot aggregate and has been deprecated. Use Series.transform to keep behavior unchanged.",
            category=FutureWarning,
        )
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    checkpoint = torch.load(str(resolved_checkpoint), weights_only=False, map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    uid_series = torch.unique(test_data.dataset.inter_feat[dataset.uid_field])
    uid_array = uid_series.detach().cpu().numpy().astype(np.int64)
    topk_score, topk_iid_list = full_sort_topk(uid_array, model, test_data, k=k, device=model.device)

    user_ids = dataset.id2token(dataset.uid_field, uid_array)
    candidate_ids = topk_iid_list.detach().cpu().numpy().astype(np.int64)
    score_matrix = topk_score.detach().cpu().numpy().astype(np.float32)

    raw_item_ids = dataset.id2token(dataset.iid_field, candidate_ids.reshape(-1)).reshape(candidate_ids.shape)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as handle:
        for index, raw_user_id in enumerate(user_ids):
            handle.write(
                json.dumps(
                    {
                        "user_id": str(raw_user_id),
                        "candidates": [
                            {"item_id": str(item_id), "base_score": float(score)}
                            for item_id, score in zip(raw_item_ids[index].tolist(), score_matrix[index].tolist())
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
