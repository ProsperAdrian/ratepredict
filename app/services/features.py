from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBuildResult:
    features: pd.DataFrame
    merged: pd.DataFrame


class PublicFeatureBuilder:
    """Rebuild the exact 42-feature runtime set from the final training notebook."""

    def build(
        self,
        export_frame: pd.DataFrame,
        external_daily: pd.DataFrame,
        feature_columns: list[str],
    ) -> FeatureBuildResult:
        external = external_daily.copy().sort_index()
        external_2h = external.resample("2h").ffill()

        df = export_frame.join(external_2h, how="left")
        ext_cols = list(external.columns)
        df[ext_cols] = df[ext_cols].ffill()

        feat = pd.DataFrame(index=df.index)

        for periods, label in [(1, "2h"), (4, "8h"), (12, "24h"), (36, "3d"), (84, "7d")]:
            feat[f"return_{label}"] = df["close"].pct_change(periods)

        sma_5d = df["close"].rolling(5 * 12).mean()
        sma_20d = df["close"].rolling(20 * 12).mean()
        feat["sma_5d_20d_cross"] = (sma_5d - sma_20d) / sma_20d.replace(0, np.nan)

        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        feat["ema_macd"] = (ema_12 - ema_26) / ema_26.replace(0, np.nan)

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        feat["rsi_14"] = 100 - (100 / (1 + rs))

        if "brent" in df.columns:
            feat["brent_roc_3d"] = df["brent"].pct_change(36)
            feat["brent_roc_7d"] = df["brent"].pct_change(84)
        if "dxy" in df.columns:
            feat["dxy_roc_3d"] = df["dxy"].pct_change(36)
            feat["dxy_roc_7d"] = df["dxy"].pct_change(84)

        if "btc_premium" not in df.columns and {"implied_btcusd_quidax", "btcusd_global"}.issubset(df.columns):
            df["btc_premium"] = (df["implied_btcusd_quidax"] / df["btcusd_global"]) - 1

        if "btc_premium" in df.columns:
            feat["btc_premium"] = df["btc_premium"]
            feat["btc_premium_ma_12"] = df["btc_premium"].rolling(12).mean()
            feat["btc_premium_std_12"] = df["btc_premium"].rolling(12).std()
            feat["btc_premium_zscore"] = (
                (df["btc_premium"] - df["btc_premium"].rolling(60).mean())
                / df["btc_premium"].rolling(60).std().replace(0, np.nan)
            )

        if {"btcusd_global", "implied_btcusd_quidax"}.issubset(df.columns):
            feat["quidax_btc_corr_24h"] = df["implied_btcusd_quidax"].rolling(12).corr(df["btcusd_global"])

        if "usdngn_official" in df.columns:
            feat["parallel_vs_official"] = (
                (df["close"] - df["usdngn_official"]) / df["usdngn_official"].replace(0, np.nan)
            )
            feat["parallel_vs_official_zscore"] = (
                (feat["parallel_vs_official"] - feat["parallel_vs_official"].rolling(360).mean())
                / feat["parallel_vs_official"].rolling(360).std().replace(0, np.nan)
            )
            feat["parallel_vs_official_roc_24h"] = feat["parallel_vs_official"].diff(12)

        returns_2h = df["close"].pct_change()
        feat["realized_vol_24h"] = returns_2h.rolling(12).std()
        feat["realized_vol_7d"] = returns_2h.rolling(84).std()
        feat["vol_ratio"] = feat["realized_vol_24h"] / feat["realized_vol_7d"].replace(0, np.nan)

        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        feat["atr_14"] = tr.rolling(14).mean()
        feat["atr_pct"] = feat["atr_14"] / df["close"].replace(0, np.nan)

        if "vix" in df.columns:
            feat["vix"] = df["vix"]
            feat["vix_roc_7d"] = df["vix"].pct_change(84)

        feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        feat["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        feat["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
        feat["is_month_end"] = (df.index.day >= 25).astype(float)
        feat["is_weekend_adjacent"] = df.index.dayofweek.isin([0, 4]).astype(float)

        for col in ["usdzar", "usdghs", "usdkes"]:
            if col in df.columns:
                feat[f"{col}_roc_1d"] = df[col].pct_change(12)
                feat[f"{col}_roc_7d"] = df[col].pct_change(84)

        africa_1d_cols = [column for column in feat.columns if column.endswith("_roc_1d") and column.startswith("usd")]
        if africa_1d_cols:
            feat["africa_fx_composite"] = feat[africa_1d_cols].mean(axis=1)

        feat = feat.replace([np.inf, -np.inf], np.nan)
        null_pct = feat.isnull().mean()
        drop_cols = [column for column in null_pct[null_pct > 0.50].index.tolist() if column not in feature_columns]
        if drop_cols:
            feat = feat.drop(columns=drop_cols)

        missing = [column for column in feature_columns if column not in feat.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")

        feat[feature_columns] = feat[feature_columns].ffill().fillna(0)
        return FeatureBuildResult(features=feat[feature_columns], merged=df)
