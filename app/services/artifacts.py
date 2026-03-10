from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.config import Settings


@dataclass(frozen=True)
class ArtifactBundle:
    xgb_model: object
    lgbm_model: object
    ridge_model: object
    scaler: object
    feature_columns: list[str]
    metadata: dict
    feature_importance: pd.DataFrame
    cv_results: pd.DataFrame


class ArtifactLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load(self) -> ArtifactBundle:
        artifacts_dir = self.settings.artifacts_dir
        with (artifacts_dir / "xgb_model.pkl").open("rb") as handle:
            xgb_model = pickle.load(handle)
        with (artifacts_dir / "lgbm_model.pkl").open("rb") as handle:
            lgbm_model = pickle.load(handle)
        with (artifacts_dir / "ridge_model.pkl").open("rb") as handle:
            ridge_model = pickle.load(handle)
        with (artifacts_dir / "scaler.pkl").open("rb") as handle:
            scaler = pickle.load(handle)
        feature_columns = json.loads((artifacts_dir / "feature_cols.json").read_text())
        metadata = json.loads((artifacts_dir / "model_metadata.json").read_text())
        feature_importance = pd.read_csv(artifacts_dir / "feature_importance.csv")
        feature_importance.columns = ["feature", "importance"]
        cv_results = pd.read_csv(artifacts_dir / "cv_results.csv")
        return ArtifactBundle(
            xgb_model=xgb_model,
            lgbm_model=lgbm_model,
            ridge_model=ridge_model,
            scaler=scaler,
            feature_columns=feature_columns,
            metadata=metadata,
            feature_importance=feature_importance,
            cv_results=cv_results,
        )


class ExportLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def latest_export_path(self) -> Path:
        runtime_path = self.settings.data_dir / self.settings.runtime_bars_filename
        if runtime_path.exists():
            return runtime_path
        candidates = sorted(self.settings.data_dir.glob(self.settings.export_glob))
        if not candidates:
            candidates = sorted(self.settings.artifacts_dir.glob(self.settings.export_glob))
        if not candidates:
            raise FileNotFoundError("No export CSV matching the configured pattern was found.")
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def load_latest(self) -> pd.DataFrame:
        path = self.latest_export_path()
        frame = pd.read_csv(path)
        frame["bucket_2h"] = pd.to_datetime(frame["bucket_2h"], utc=True)
        frame = frame.sort_values("bucket_2h").set_index("bucket_2h")
        frame.index.name = "bucket_2h"
        return frame
