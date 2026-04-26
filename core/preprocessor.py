from __future__ import annotations
import re
import pandas as pd

IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
PORT_RE = re.compile(r":\d+\b")
BLOCK_RE = re.compile(r"blk_-?\d+")
NUM_RE = re.compile(r"\b\d+\b")
PATH_RE = re.compile(r"/[\w\-/\.]+")
HEX_RE = re.compile(r"0x[0-9a-fA-F]+")

def normalize_hdfs_message(message: str) -> str:
    text = str(message)
    text = HEX_RE.sub("<HEX>", text)
    text = BLOCK_RE.sub("<BLOCK>", text)
    text = IP_RE.sub("<IP>", text)
    text = PORT_RE.sub(":<PORT>", text)
    text = PATH_RE.sub("<PATH>", text)
    text = NUM_RE.sub("<NUM>", text)
    return text.strip()


def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    df["level"] = df["level"].fillna("UNKNOWN").astype(str).str.upper()
    df["component"] = df["component"].fillna("UNKNOWN").astype(str)
    df["message"] = df["message"].fillna("").astype(str)
    df["block_id"] = df["block_id"].fillna("NO_BLOCK").astype(str)
    df["pid"] = pd.to_numeric(df["pid"], errors="coerce").fillna(0)

    level_map = {"UNKNOWN": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "FATAL": 4}
    df["level_code"] = df["level"].map(level_map).fillna(0).astype(int)

    df["template"] = df["message"].apply(normalize_hdfs_message)
    tpl_lower = df["template"].str.lower()
    msg_lower = df["message"].str.lower()

    df["has_exception"] = msg_lower.str.contains(r"\bexception\b", regex=True).astype(int)
    df["has_error_word"] = msg_lower.str.contains(
        r"\berror\b|\bfailed\b|\bfailure\b|\binvalid\b|\bcorrupt\b", regex=True
    ).astype(int)

    df["is_served_block"] = tpl_lower.str.contains(r"served block", regex=True).astype(int)
    df["is_receiving_block"] = tpl_lower.str.contains(r"receiving block", regex=True).astype(int)
    df["is_received_block"] = tpl_lower.str.contains(r"received block", regex=True).astype(int)
    df["is_terminating"] = tpl_lower.str.contains(r"terminating", regex=True).astype(int)
    df["is_addstoredblock"] = tpl_lower.str.contains(r"addstoredblock", regex=True).astype(int)
    df["is_allocateblock"] = tpl_lower.str.contains(r"allocateblock", regex=True).astype(int)
    df["is_verification_success"] = tpl_lower.str.contains(r"verification succeeded", regex=True).astype(int)
    df["is_deleting_block"] = tpl_lower.str.contains(r"deleting block", regex=True).astype(int)

 
    template_freq = df["template"].value_counts()
    component_freq = df["component"].value_counts()

    df["template_freq"] = df["template"].map(template_freq)
    df["component_freq"] = df["component"].map(component_freq)
    df["template_rarity"] = 1.0 / df["template_freq"].clip(lower=1)

    grouped = df.groupby("block_id", dropna=False)

    df_block = grouped.agg(
        line_count=("message", "size"),
        unique_templates=("template", "nunique"),
        unique_components=("component", "nunique"),
        avg_level=("level_code", "mean"),
        max_level=("level_code", "max"),
        warn_error_count=("level", lambda s: s.isin(["WARN", "ERROR", "FATAL"]).sum()),
        exception_count=("has_exception", "sum"),
        error_word_count=("has_error_word", "sum"),
        avg_template_rarity=("template_rarity", "mean"),
        max_template_rarity=("template_rarity", "max"),
        served_block_count=("is_served_block", "sum"),
        receiving_block_count=("is_receiving_block", "sum"),
        received_block_count=("is_received_block", "sum"),
        terminating_count=("is_terminating", "sum"),
        addstoredblock_count=("is_addstoredblock", "sum"),
        allocateblock_count=("is_allocateblock", "sum"),
        verification_success_count=("is_verification_success", "sum"),
        deleting_block_count=("is_deleting_block", "sum"),
    ).reset_index()

 
    df_block["warn_error_ratio"] = df_block["warn_error_count"] / df_block["line_count"].clip(lower=1)
    df_block["exception_ratio"] = df_block["exception_count"] / df_block["line_count"].clip(lower=1)
    df_block["error_word_ratio"] = df_block["error_word_count"] / df_block["line_count"].clip(lower=1)

    df_block["missing_receive_after_allocate"] = (
        (df_block["allocateblock_count"] > 0) & (df_block["received_block_count"] == 0)
    ).astype(int)

    df_block["termination_without_success"] = (
        (df_block["terminating_count"] > 0) & (df_block["verification_success_count"] == 0)
    ).astype(int)

    feature_cols = [
        "line_count",
        "unique_templates",
        "unique_components",
        "avg_level",
        "max_level",
        "warn_error_count",
        "exception_count",
        "error_word_count",
        "avg_template_rarity",
        "max_template_rarity",
        "warn_error_ratio",
        "exception_ratio",
        "error_word_ratio",
        "served_block_count",
        "receiving_block_count",
        "received_block_count",
        "terminating_count",
        "addstoredblock_count",
        "allocateblock_count",
        "verification_success_count",
        "deleting_block_count",
        "missing_receive_after_allocate",
        "termination_without_success",
    ]

    return df_block[feature_cols].copy(), df_block