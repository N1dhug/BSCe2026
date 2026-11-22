"""Smoke test: all retained Chapter 3–5 executables have valid import dependencies."""

import importlib

REPORT_MODULES = (
    "src.exe.data.baseline_diagnostics",
    "src.exe.data.season_overview",
    "src.exe.data.weeks_overview",
    "src.exe.run_model.bm_1.evaluate_fixed_beta_samples",
    "src.exe.run_model.bm_1.fit_single_day",
    "src.exe.run_model.wm_4.fit_single_day",
    "src.exe.run_model.wm_4.fit_moving_horizon_diffusion",
    "src.exe.run_model.wm_4.plot_baseline_moving_horizon_quality",
    "src.exe.run_model.wm_4.plot_corresponding_interval_fit",
    "src.exe.run_model.wm_4.plot_diffusion_demand_std",
    "src.exe.run_model.wm_4.plot_moving_horizon_predictions",
    "src.exe.run_model.wm_4.plot_moving_horizon_quality",
    "src.exe.run_model.wm_4.plot_selected_sample_oracle_nll",
    "src.exe.run_model.wm_4.report_best_k24",
    "src.exe.run_model.wm_4.selected_bm_dates_arm",
    "src.exe.run_model.wm_4.selected_sample_nll",
    "src.exe.run_model.wm_4.selected_sample_quality",
    "src.exe.run_model.wm_4.sweep_weekday_horizon",
)


def test_all_report_entrypoints_import():
    for name in REPORT_MODULES:
        module=importlib.import_module(name)
        assert hasattr(module, "main") or hasattr(module, "run"), name


def test_report_model_and_estimator_exports():
    from src.model import bm_1, wm_4
    from src.params import bm_1_fixed_beta_params, joint_qp_params, optimized_qp_params
    assert bm_1 is not None and wm_4 is not None
    assert bm_1_fixed_beta_params is not None
    assert joint_qp_params is not None and optimized_qp_params is not None
