from pathlib import Path
import pandas as pd
import yaml

from cleanbars.normalize import validate_panel, save_panel_parquet
from cleanbars.calendars import (
    build_session_completeness_audit, 
    filter_complete_sessions,
    build_calendar_audit,
    expected_sessions
)
from cleanbars.validate import (
    build_vendor_comparison,
    build_structural_checks,
    summarize_checks
)

# Safely import jump forensics and jump checks
try:
    from cleanbars.jumps import build_jump_forensics, build_jump_checks
except ImportError:
    try:
        from cleanbars.forensics import build_jump_forensics, build_jump_checks
    except ImportError:
        from cleanbars.validate import build_jump_forensics, build_jump_checks

ROOT = Path(__file__).resolve().parents[1]

# Input Paths
YF_INTERIM = ROOT / "data/interim/yfinance_daily.parquet"
AV_INTERIM = ROOT / "data/interim/alpha_vantage_daily.parquet"

# Output Paths
YF_COMPLETE = ROOT / "data/processed/yfinance_daily_complete.parquet"
AV_COMPLETE = ROOT / "data/processed/alpha_vantage_daily_complete.parquet"
VENDOR_COMPARISON = ROOT / "data/processed/vendor_comparison.parquet"
VALIDATION_DIR = ROOT / "reports/validation"

def main():
    # --- 1. Setup & Load Configurations ---
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    YF_COMPLETE.parent.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    universe = config["universe"]

    # --- 2. Load and Validate Interim Panels ---
    yf_panel = pd.read_parquet(YF_INTERIM)
    av_panel = pd.read_parquet(AV_INTERIM)

    validate_panel(yf_panel)
    validate_panel(av_panel)

    # --- 3. Build Session-Completeness Audits ---
    yf_audit = build_session_completeness_audit(yf_panel)
    av_audit = build_session_completeness_audit(av_panel)

    # --- 4. Filter Complete Sessions ---
    yf_complete = filter_complete_sessions(yf_panel, yf_audit)
    av_complete = filter_complete_sessions(av_panel, av_audit)

    # --- 5. Save Complete-Session Panels & Audit Parquets ---
    save_panel_parquet(yf_complete, YF_COMPLETE)
    save_panel_parquet(av_complete, AV_COMPLETE)
    
    yf_audit.to_parquet(VALIDATION_DIR / "yfinance_session_completeness.parquet", engine="pyarrow", compression="zstd")
    av_audit.to_parquet(VALIDATION_DIR / "alpha_vantage_session_completeness.parquet", engine="pyarrow", compression="zstd")

    # --- 6. Build and Save Vendor Comparison ---
    comparison = build_vendor_comparison(yf_complete, av_complete)
    comparison.to_parquet(VENDOR_COMPARISON, engine="pyarrow", compression="zstd")

    # --- 7. Build Compact CSV Summaries ---
    
    # 7a. Structural Checks
    yf_struct = summarize_checks(build_structural_checks(yf_complete)).reset_index()
    yf_struct.insert(0, "vendor", "yfinance")
    
    av_struct = summarize_checks(build_structural_checks(av_complete)).reset_index()
    av_struct.insert(0, "vendor", "alpha_vantage")
    
    structural_summary = pd.concat([yf_struct, av_struct], ignore_index=True)
    cols = ["vendor", "asset_id"] + [c for c in structural_summary.columns if c not in ["vendor", "asset_id"]]
    structural_summary = structural_summary[cols]

    # 7b. Calendar Audits
    yf_max = yf_complete.index.get_level_values("date").max()
    yf_sessions = expected_sessions("1970-01-01", yf_max)
    yf_cal = build_calendar_audit(yf_complete, yf_sessions, audit_start="1970-01-01")
    
    av_max = av_complete.index.get_level_values("date").max()
    av_sessions = expected_sessions("1970-01-01", av_max)
    av_cal = build_calendar_audit(av_complete, av_sessions, audit_start="1970-01-01")
    
    yf_cal_agg = yf_cal.groupby("asset_id")[["missing_vendor_row", "unexpected_session"]].sum().astype(int)
    av_cal_agg = av_cal.groupby("asset_id")[["missing_vendor_row", "unexpected_session"]].sum().astype(int)
    
    calendar_summary = pd.DataFrame({
        "yfinance_missing_rows": yf_cal_agg["missing_vendor_row"],
        "yfinance_unexpected_sessions": yf_cal_agg["unexpected_session"],
        "alpha_vantage_missing_rows": av_cal_agg["missing_vendor_row"],
        "alpha_vantage_unexpected_sessions": av_cal_agg["unexpected_session"]
    }).reindex(universe, fill_value=0)
    
    calendar_summary.index.name = "asset_id"
    calendar_summary = calendar_summary.reset_index()

    # 7c. Incomplete-Session Counts
    incomplete_summary = pd.DataFrame({
        "yfinance_incomplete": yf_audit["potentially_incomplete"].groupby(level="asset_id").sum().astype(int),
        "alpha_vantage_incomplete": av_audit["potentially_incomplete"].groupby(level="asset_id").sum().astype(int),
    }).reindex(universe, fill_value=0).reset_index()

    # 7d. Cross-Vendor Differences
    overlap = comparison["yfinance_observed"] & comparison["alpha_vantage_observed"]
    yf_only = comparison["yfinance_observed"] & ~comparison["alpha_vantage_observed"]
    av_only = comparison["alpha_vantage_observed"] & ~comparison["yfinance_observed"]
    
    shared = comparison.loc[overlap]
    
    vendor_summary = pd.DataFrame({
        "overlap_rows": overlap.groupby(level="asset_id").sum(),
        "yfinance_only_rows": yf_only.groupby(level="asset_id").sum(),
        "alpha_vantage_only_rows": av_only.groupby(level="asset_id").sum(),
        "median_close_rel_diff": shared.groupby(level="asset_id")["close_rel_diff"].median(),
        "max_close_rel_diff": shared.groupby(level="asset_id")["close_rel_diff"].max(),
        "median_volume_rel_diff": shared.groupby(level="asset_id")["volume_rel_diff"].median(),
        "max_volume_rel_diff": shared.groupby(level="asset_id")["volume_rel_diff"].max(),
    }).reindex(universe).fillna(0)
    
    for col in ["overlap_rows", "yfinance_only_rows", "alpha_vantage_only_rows"]:
        vendor_summary[col] = vendor_summary[col].astype(int)
        
    vendor_summary.index.name = "asset_id"
    vendor_summary = vendor_summary.reset_index()

    # 7e. Suspicious-Jump Counts & 8. Detailed Jump Forensics
    yf_jumps = build_jump_forensics(yf_complete)
    av_jumps = build_jump_forensics(av_complete)
    
    yf_jumps["vendor"] = "yfinance"
    av_jumps["vendor"] = "alpha_vantage"
    all_jumps = pd.concat([yf_jumps, av_jumps])
    all_jumps.to_parquet(VALIDATION_DIR / "jump_forensics.parquet", engine="pyarrow", compression="zstd")

    yf_jump_counts = (
        build_jump_checks(yf_complete)["suspicious_jump"]
        .groupby(level="asset_id")
        .sum()
        .reindex(universe, fill_value=0)
        .astype(int)
    )
    av_jump_counts = (
        build_jump_checks(av_complete)["suspicious_jump"]
        .groupby(level="asset_id")
        .sum()
        .reindex(universe, fill_value=0)
        .astype(int)
    )

    jump_summary = pd.DataFrame({
        "yfinance_jumps": yf_jump_counts,
        "alpha_vantage_jumps": av_jump_counts
    })
    jump_summary.index.name = "asset_id"
    jump_summary = jump_summary.reset_index()

    # --- 9. Defensive Table Assertions ---
    assert not jump_summary.isna().any().any(), "Jump summary contains missing values."
    assert len(jump_summary) == len(universe), "Jump summary length does not match universe."
    assert {
        "overlap_rows",
        "yfinance_only_rows",
        "alpha_vantage_only_rows",
    }.issubset(vendor_summary.columns), "Vendor summary missing requested overlap columns."
    assert not vendor_summary.isna().any().any(), "Vendor summary contains missing values."
    assert set(calendar_summary["asset_id"]) == set(universe), "Calendar summary is missing assets from the universe."
    assert {"vendor", "asset_id"}.issubset(structural_summary.columns), "Structural summary missing critical ID columns."
    assert len(structural_summary) == 2 * len(universe), "Structural summary has incorrect number of rows."
    assert set(calendar_summary.columns) != set(jump_summary.columns), "Calendar and Jump summaries share identical columns (copy error)."

    # --- Save CSVs ---
    structural_summary.to_csv(VALIDATION_DIR / "structural_summary.csv", index=False)
    calendar_summary.to_csv(VALIDATION_DIR / "calendar_summary.csv", index=False)
    incomplete_summary.to_csv(VALIDATION_DIR / "session_completeness_summary.csv", index=False)
    vendor_summary.to_csv(VALIDATION_DIR / "vendor_comparison_summary.csv", index=False)
    jump_summary.to_csv(VALIDATION_DIR / "jump_summary.csv", index=False)

    # --- Final Dataset Assertions ---
    validate_panel(yf_complete)
    validate_panel(av_complete)
    assert yf_complete.index.is_unique, "YF complete index is not unique."
    assert av_complete.index.is_unique, "AV complete index is not unique."
    assert comparison.index.is_unique, "Vendor comparison index is not unique."

    assert not build_session_completeness_audit(yf_complete)["potentially_incomplete"].any(), "Incomplete sessions remain in YF."
    assert not build_session_completeness_audit(av_complete)["potentially_incomplete"].any(), "Incomplete sessions remain in AV."

    # --- 10. Print Manifest ---
    manifest = [
        YF_COMPLETE,
        AV_COMPLETE,
        VENDOR_COMPARISON,
        VALIDATION_DIR / "yfinance_session_completeness.parquet",
        VALIDATION_DIR / "alpha_vantage_session_completeness.parquet",
        VALIDATION_DIR / "jump_forensics.parquet",
        VALIDATION_DIR / "structural_summary.csv",
        VALIDATION_DIR / "calendar_summary.csv",
        VALIDATION_DIR / "session_completeness_summary.csv",
        VALIDATION_DIR / "vendor_comparison_summary.csv",
        VALIDATION_DIR / "jump_summary.csv"
    ]
    
    print("\nRebuild complete. Generated artifacts:")
    for path in manifest:
        print(f"WROTE {path.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
