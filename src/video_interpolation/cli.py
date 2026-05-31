from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .baselines import (
    BaselineEvaluationConfig,
    BaselineEvaluationResult,
    evaluate_baselines,
    with_baseline_config_path,
    with_baseline_limit,
    with_baseline_lpips,
    with_baseline_manifest,
    with_baseline_mlflow_disabled,
    with_baseline_names,
    with_baseline_output_dir,
    with_baseline_save_predictions,
)
from .batch_inference import (
    BatchInferenceVideo,
    batch_measurements_path,
    batch_output_path,
    batch_run_name,
    discover_inference_videos,
    resolve_batch_target_names,
    write_batch_measurements_csv,
)
from .adapters.base import AdapterEnvironmentReport, ModelAdapterError
from .adapters.amt import AMTAdapter, AMTAdapterConfig
from .adapters.baseline import BaselineAdapter, BaselineAdapterConfig, with_baseline_adapter_name
from .adapters.ema_vfi import EMAVFIAdapter, EMAVFIAdapterConfig
from .adapters.rife import PracticalRIFEAdapter, PracticalRIFEAdapterConfig
from .amt_preflight import run_amt_s_preflight
from .data.datasets import UniversalTripletDataset
from .data.indexing import (
    GlobalIndexConfig,
    GlobalIndexResult,
    SourceIndexResult,
    VimeoTripletIndexConfig,
    build_global_sequence_index,
    build_vimeo_triplet_index,
    with_global_limit,
    with_global_output_path,
    with_limit as with_vimeo_limit,
    with_output_path,
    with_source_index_paths,
)
from .data.preprocessing import (
    PreprocessResult,
    VideoPreprocessConfig,
    preprocess_videos,
    with_debug_overrides,
)
from .data.versioning import (
    DatasetVersionConfig,
    DatasetVersionResult,
    build_dataset_version,
    with_dataset_limit_sequences,
    with_dataset_output_root,
    with_dataset_source_index,
    with_dataset_version_id,
)
from .ema_preflight import PreflightReport, run_ema_vfi_small_preflight
from .inference import (
    VideoInferenceConfig,
    VideoInferenceResult,
    run_ema_video_inference,
    run_video_inference,
    with_inference_checkpoint,
    with_inference_codec,
    with_inference_config_path,
    with_inference_input,
    with_inference_limit,
    with_inference_mlflow_disabled,
    with_inference_output,
)
from .mlflow import MlflowLoggingError, MlflowRunConfig, MlflowSmokeResult, run_mlflow_smoke
from .rife_preflight import run_practical_rife_preflight
from .settings import load_settings
from .training import (
    EMATrainingConfig,
    EMATrainingResult,
    run_ema_training,
    with_training_checkpoint,
    with_training_config_path,
    with_training_limit,
    with_training_mlflow_disabled,
    with_training_output_dir,
    with_training_sample_limits,
)
from .validation import (
    CandidateValidationConfig,
    CandidateValidationResult,
    validate_candidate,
    validate_candidate_with_adapter,
    with_validation_checkpoint,
    with_validation_config_path,
    with_validation_limit,
    with_validation_lpips,
    with_validation_manifest,
    with_validation_mlflow_disabled,
    with_validation_output_dir,
)


app = typer.Typer(
    help="Developer CLI for the local video interpolation ML core.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Data preprocessing and source indexing commands.")
baseline_app = typer.Typer(help="Baseline evaluation commands.")
mlflow_app = typer.Typer(help="MLflow infrastructure and logging checks.")
ema_app = typer.Typer(help="EMA-VFI-small adapter, inference, training, and validation commands.")
amt_app = typer.Typer(help="AMT-S adapter, inference, and validation commands.")
rife_app = typer.Typer(help="Practical-RIFE adapter, inference, and validation commands.")
console = Console()

BATCH_TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "all": (
        "practical_rife_v4_25",
        "amt_s",
        "ema_vfi_small",
        "baseline_duplicate_left",
        "baseline_blend",
        "baseline_farneback",
    ),
    "models": ("practical_rife_v4_25", "amt_s", "ema_vfi_small"),
    "model": ("practical_rife_v4_25", "amt_s", "ema_vfi_small"),
    "neural": ("practical_rife_v4_25", "amt_s", "ema_vfi_small"),
    "baselines": ("baseline_duplicate_left", "baseline_blend", "baseline_farneback"),
    "baseline": ("baseline_duplicate_left", "baseline_blend", "baseline_farneback"),
    "rife": ("practical_rife_v4_25",),
    "amt": ("amt_s",),
    "ema": ("ema_vfi_small",),
    "duplicate_left": ("baseline_duplicate_left",),
    "blend": ("baseline_blend",),
    "farneback": ("baseline_farneback",),
}


@dataclass(frozen=True)
class _InferenceTarget:
    name: str
    label: str
    config_path: Path
    output_group: Path
    model_config_factory: Callable[[dict[str, Any] | None], Any]
    adapter_factory: Callable[[Any, Any], Any]
    mlflow_mode: str
    baseline_name: str | None = None


@dataclass(frozen=True)
class _BatchInferenceRecord:
    target_name: str
    input_video: Path
    relative_input_video: Path
    output_video: Path | None
    status: str
    pairs_processed: int | None = None
    frames_written: int | None = None
    input_fps: float | None = None
    output_fps: float | None = None
    model_inference_elapsed_sec: float | None = None
    total_elapsed_sec: float | None = None
    model_pairs_per_sec: float | None = None
    total_pairs_per_sec: float | None = None
    audio_streams_available: int | None = None
    audio_streams_preserved: int | None = None
    mlflow_run_id: str | None = None
    error: str | None = None


@app.command("show-settings")
def show_settings() -> None:
    """Show resolved runtime settings loaded from environment and .env."""
    settings = load_settings()
    table = Table(title="Video Interpolation Settings")
    table.add_column("Name")
    table.add_column("Value")
    table.add_row("DATASET_ROOT", str(settings.dataset_root_abs))
    table.add_row("MODEL_REPOS_ROOT", str(settings.model_repos_root_abs))
    table.add_row("MODEL_WEIGHTS_ROOT", str(settings.model_weights_root_abs))
    table.add_row("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
    console.print(table)


@app.command("infer-all-videos")
def infer_all_videos(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        help="Directory containing input videos. Required; no default input directory is used.",
    ),
    output_root: Path = typer.Option(
        Path("outputs/inference"),
        "--output-root",
        help="Root directory for output videos.",
    ),
    limit_videos: int | None = typer.Option(
        None,
        "--limit-videos",
        min=1,
        help="Optional maximum number of discovered videos to process.",
    ),
    limit_pairs: int | None = typer.Option(
        None,
        "--limit-pairs",
        min=1,
        help="Optional cap on neighboring frame pairs per video for smoke runs.",
    ),
    codec: str | None = typer.Option(
        None,
        "--codec",
        help="Override FFmpeg/PyAV encoder for all outputs, for example libx264 or h264_nvenc.",
    ),
    target_selection: list[str] | None = typer.Option(
        None,
        "--target",
        "--method",
        help=(
            "Inference target or subgroup to run. Repeat for multiple values. "
            "Examples: ema_vfi_small, amt_s, practical_rife_v4_25, blend, farneback, models, baselines."
        ),
    ),
    disable_mlflow: bool = typer.Option(
        False,
        "--disable-mlflow",
        help="Skip MLflow logging for every inference run.",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help="Continue remaining videos/targets after a failed run, or stop at the first failure.",
    ),
) -> None:
    """Run every Stage 1 model and baseline method over every video in a directory."""
    settings = load_settings()
    resolved_input_dir = settings.resolve_path(input_dir)
    resolved_output_root = settings.resolve_path(output_root)
    try:
        videos = discover_inference_videos(resolved_input_dir, limit_videos=limit_videos)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not videos:
        console.print(f"[red]No supported input videos found under: {resolved_input_dir}[/red]")
        raise typer.Exit(code=1)

    try:
        targets = _selected_inference_targets(_inference_targets(), target_selection)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        "[bold]Running selected Stage 1 video inference targets[/bold]\n"
        f"videos={len(videos)} targets={len(targets)} output_root={resolved_output_root}"
    )
    records: list[_BatchInferenceRecord] = []

    with _progress() as progress:
        jobs_task = progress.add_task("Inference jobs", total=len(videos) * len(targets))
        pairs_task = progress.add_task("Current frame pairs", total=None, visible=False)

        for target in targets:
            try:
                target_config = _load_batch_target_config(
                    target,
                    limit_pairs=limit_pairs,
                    codec=codec,
                    disable_mlflow=disable_mlflow,
                )
                adapter = target.adapter_factory(target_config.model, settings)
                adapter.load_checkpoint()
            except (ModelAdapterError, FileNotFoundError, ValueError) as exc:
                console.print(f"[red]{target.label} setup failed:[/red] {exc}")
                for video in videos:
                    records.append(
                        _BatchInferenceRecord(
                            target_name=target.name,
                            input_video=video.input_path,
                            relative_input_video=video.relative_path,
                            output_video=None,
                            status="failed",
                            error=str(exc),
                        )
                    )
                    progress.advance(jobs_task)
                if not continue_on_error:
                    measurement_paths = _write_batch_measurement_files(records, targets, resolved_output_root)
                    _print_batch_inference_summary(records, measurement_paths)
                    raise typer.Exit(code=1) from exc
                continue

            try:
                for video in videos:
                    run_config = _batch_video_config(target_config, target, video, resolved_output_root)

                    def on_progress(event: str, payload: dict[str, object]) -> None:
                        if event == "start":
                            total = int(payload.get("total") or 0)
                            progress.update(
                                pairs_task,
                                description=f"{target.name}: {video.relative_path}",
                                total=total if total > 0 else None,
                                completed=0,
                                visible=True,
                            )
                        elif event == "encoding_start":
                            console.print(
                                "[dim]encoding "
                                f"{payload['output_path']} "
                                f"codec={payload['codec']} "
                                f"fps={float(payload['output_fps']):.4f} "
                                f"audio={payload['audio_streams_to_preserve']}/"
                                f"{payload['audio_streams_available']}[/dim]"
                            )
                        elif event == "pair_advanced":
                            progress.advance(pairs_task)

                    try:
                        result = run_video_inference(
                            run_config,
                            adapter_factory=target.adapter_factory,
                            settings=settings,
                            adapter=adapter,
                            progress_callback=on_progress,
                            mlflow_mode=target.mlflow_mode,
                        )
                        records.append(
                            _BatchInferenceRecord(
                                target_name=target.name,
                                input_video=video.input_path,
                                relative_input_video=video.relative_path,
                                output_video=result.output_path,
                                status="ok",
                                pairs_processed=result.pairs_processed,
                                frames_written=result.frames_written,
                                input_fps=result.input_fps,
                                output_fps=result.output_fps,
                                model_inference_elapsed_sec=result.model_inference_elapsed_sec,
                                total_elapsed_sec=result.total_elapsed_sec,
                                model_pairs_per_sec=result.model_pairs_per_sec,
                                total_pairs_per_sec=result.total_pairs_per_sec,
                                audio_streams_available=result.audio_streams_available,
                                audio_streams_preserved=result.audio_streams_preserved,
                                mlflow_run_id=result.mlflow_run_id,
                            )
                        )
                    except (MlflowLoggingError, ModelAdapterError, FileNotFoundError, ValueError) as exc:
                        records.append(
                            _BatchInferenceRecord(
                                target_name=target.name,
                                input_video=video.input_path,
                                relative_input_video=video.relative_path,
                                output_video=run_config.output_path,
                                status="failed",
                                error=str(exc),
                            )
                        )
                        console.print(f"[red]{target.label} failed on {video.relative_path}:[/red] {exc}")
                        if not continue_on_error:
                            measurement_paths = _write_batch_measurement_files(records, targets, resolved_output_root)
                            _print_batch_inference_summary(records, measurement_paths)
                            raise typer.Exit(code=1) from exc
                    finally:
                        progress.advance(jobs_task)
                        progress.update(pairs_task, visible=False)
            finally:
                adapter.close()

    measurement_paths = _write_batch_measurement_files(records, targets, resolved_output_root)
    _print_batch_inference_summary(records, measurement_paths)
    if any(record.status != "ok" for record in records):
        raise typer.Exit(code=1)


def _print_preflight_report(report: PreflightReport, title: str = "EMA-VFI-small preflight") -> None:
    table = Table(title=f"{title}: {report.status}")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in report.checks:
        table.add_row(check.name, check.status, check.detail)
    console.print(table)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    config = OmegaConf.load(path)
    data = OmegaConf.to_container(config, resolve=True)
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Config must contain a mapping: {path}")
    return data


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


@app.command("ema-preflight")
def ema_preflight(
    fail_on_blocked: bool = typer.Option(
        False,
        "--fail-on-blocked/--no-fail-on-blocked",
        help="Return a non-zero exit code when compatibility is blocked.",
    ),
) -> None:
    """Check whether EMA-VFI-small can be imported, initialized, and loaded."""
    report = run_ema_vfi_small_preflight()
    _print_preflight_report(report)

    if report.status == "failed" or (report.status == "blocked" and fail_on_blocked):
        raise typer.Exit(code=1)


@app.command("amt-preflight")
def amt_preflight(
    config: Path = typer.Option(
        Path("configs/models/amt_s.yaml"),
        "--config",
        help="YAML config with AMT-S adapter parameters.",
    ),
    fail_on_blocked: bool = typer.Option(
        False,
        "--fail-on-blocked/--no-fail-on-blocked",
        help="Return a non-zero exit code when compatibility is blocked.",
    ),
) -> None:
    """Check whether AMT-S can be imported, initialized, and loaded."""
    adapter_config = AMTAdapterConfig.from_mapping(_load_yaml_mapping(config))
    report = run_amt_s_preflight(adapter_config)
    _print_preflight_report(report, title="AMT-S preflight")

    if report.status == "failed" or (report.status == "blocked" and fail_on_blocked):
        raise typer.Exit(code=1)


@app.command("rife-preflight")
def rife_preflight(
    config: Path = typer.Option(
        Path("configs/models/practical_rife_v4_25.yaml"),
        "--config",
        help="YAML config with Practical-RIFE adapter parameters.",
    ),
    fail_on_blocked: bool = typer.Option(
        False,
        "--fail-on-blocked/--no-fail-on-blocked",
        help="Return a non-zero exit code when compatibility is blocked.",
    ),
) -> None:
    """Check whether Practical-RIFE can be imported, initialized, and loaded."""
    adapter_config = PracticalRIFEAdapterConfig.from_mapping(_load_yaml_mapping(config))
    report = run_practical_rife_preflight(adapter_config)
    _print_preflight_report(report, title="Practical-RIFE preflight")

    if report.status == "failed" or (report.status == "blocked" and fail_on_blocked):
        raise typer.Exit(code=1)


@ema_app.command("adapter-check")
def ema_adapter_check(
    config: Path = typer.Option(
        Path("configs/models/ema_vfi_small.yaml"),
        "--config",
        help="YAML config with EMA-VFI-small adapter parameters.",
    ),
    fail_on_blocked: bool = typer.Option(
        False,
        "--fail-on-blocked/--no-fail-on-blocked",
        help="Return a non-zero exit code when compatibility is blocked.",
    ),
) -> None:
    """Check EMA-VFI-small adapter prerequisites without running inference."""
    adapter_config = EMAVFIAdapterConfig.from_mapping(_load_yaml_mapping(config))
    adapter = EMAVFIAdapter(adapter_config)
    report = adapter.validate_environment()
    adapter.close()
    _print_adapter_environment_report("EMA-VFI-small adapter", report)
    if report.status == "failed" or (report.status == "blocked" and fail_on_blocked):
        raise typer.Exit(code=1)


@ema_app.command("infer-video")
def ema_infer_video(
    config: Path = typer.Option(
        Path("configs/inference/ema_vfi_small_2x.yaml"),
        "--config",
        help="YAML config with EMA-VFI-small local video inference parameters.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help="Override input video path.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        help="Override output video path.",
    ),
    limit_pairs: int | None = typer.Option(
        None,
        "--limit-pairs",
        min=1,
        help="Optional cap on interpolated neighboring frame pairs for smoke runs.",
    ),
    checkpoint_path: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Override EMA checkpoint path. Relative paths resolve under MODEL_WEIGHTS_ROOT.",
    ),
    codec: str | None = typer.Option(
        None,
        "--codec",
        help="Override FFmpeg/PyAV encoder name, for example libx264 or h264_nvenc.",
    ),
    disable_mlflow: bool = typer.Option(
        False,
        "--disable-mlflow",
        help="Skip MLflow logging for a tiny local smoke run.",
    ),
) -> None:
    """Run local 2x video inference with EMA-VFI-small."""
    inference_config = VideoInferenceConfig.from_mapping(_load_yaml_mapping(config))
    inference_config = with_inference_config_path(inference_config, config)
    inference_config = with_inference_input(inference_config, input_path)
    inference_config = with_inference_output(inference_config, output_path)
    inference_config = with_inference_limit(inference_config, limit_pairs)
    inference_config = with_inference_checkpoint(inference_config, checkpoint_path)
    inference_config = with_inference_codec(inference_config, codec)
    inference_config = with_inference_mlflow_disabled(inference_config, disable_mlflow)
    console.print(f"[bold]Running EMA-VFI-small video inference:[/bold] {inference_config.input_path}")

    with _progress() as progress:
        pairs_task = progress.add_task("Frame pairs", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                total = int(payload.get("total") or 0)
                progress.update(pairs_task, total=total if total > 0 else None)
            elif event == "encoding_start":
                _print_inference_encoding_settings(payload)
            elif event == "pair_advanced":
                progress.advance(pairs_task)

        try:
            result = run_ema_video_inference(inference_config, progress_callback=on_progress)
        except (MlflowLoggingError, ModelAdapterError, FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _print_video_inference_summary(result)


@ema_app.command("finetune")
def ema_finetune(
    config: Path = typer.Option(
        Path("configs/training/ema_vfi_small_finetune.yaml"),
        "--config",
        help="YAML config with EMA-VFI-small fine-tuning parameters.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Override training output directory.",
    ),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        min=1,
        help="Override maximum optimization steps for tiny smoke runs.",
    ),
    limit_train_samples: int | None = typer.Option(
        None,
        "--limit-train-samples",
        min=1,
        help="Override the maximum number of training manifest rows to load.",
    ),
    limit_val_samples: int | None = typer.Option(
        None,
        "--limit-val-samples",
        min=1,
        help="Override the maximum number of validation manifest rows to load.",
    ),
    checkpoint_path: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Override starting checkpoint path. Relative paths resolve under MODEL_WEIGHTS_ROOT.",
    ),
    disable_mlflow: bool = typer.Option(
        False,
        "--disable-mlflow",
        help="Skip MLflow logging for a tiny local smoke run.",
    ),
) -> None:
    """Fine-tune EMA-VFI-small from local pretrained weights."""
    training_config = EMATrainingConfig.from_mapping(_load_yaml_mapping(config))
    training_config = with_training_config_path(training_config, config)
    training_config = with_training_output_dir(training_config, output_dir)
    training_config = with_training_limit(training_config, max_steps)
    training_config = with_training_sample_limits(training_config, limit_train_samples, limit_val_samples)
    training_config = with_training_checkpoint(training_config, checkpoint_path)
    training_config = with_training_mlflow_disabled(training_config, disable_mlflow)
    console.print(f"[bold]Fine-tuning EMA-VFI-small:[/bold] {training_config.train_manifest_path}")

    with _progress() as progress:
        steps_task = progress.add_task("Training steps", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                total = int(payload.get("total") or 0)
                progress.update(steps_task, total=total if total > 0 else None)
            elif event == "step_advanced":
                progress.advance(steps_task)
            elif event == "validation_done":
                console.print(
                    "[green]validation[/green] "
                    f"step={payload['step']} "
                    f"psnr={float(payload['psnr_mean']):.4f} "
                    f"ssim={float(payload['ssim_mean']):.4f}"
                )

        try:
            result = run_ema_training(training_config, progress_callback=on_progress)
        except (MlflowLoggingError, ModelAdapterError, FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _print_ema_training_summary(result)


@ema_app.command("validate-candidate")
def ema_validate_candidate(
    config: Path = typer.Option(
        Path("configs/validation/ema_vfi_small_candidate.yaml"),
        "--config",
        help="YAML config with EMA-VFI-small candidate validation parameters.",
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="Override test manifest path. Use test_all.csv, not train/val manifests.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Override candidate validation output directory.",
    ),
    limit_samples: int | None = typer.Option(
        None,
        "--limit-samples",
        min=1,
        help="Optional sample cap for smoke runs.",
    ),
    checkpoint_path: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Override candidate checkpoint path. Relative paths resolve under MODEL_WEIGHTS_ROOT.",
    ),
    no_lpips: bool = typer.Option(
        False,
        "--no-lpips",
        help="Disable LPIPS for a fast CPU smoke run.",
    ),
    disable_mlflow: bool = typer.Option(
        False,
        "--disable-mlflow",
        help="Skip MLflow logging for a tiny local smoke run.",
    ),
) -> None:
    """Validate an EMA-VFI-small checkpoint candidate on a test manifest."""
    validation_config = CandidateValidationConfig.from_mapping(_load_yaml_mapping(config))
    validation_config = with_validation_config_path(validation_config, config)
    validation_config = with_validation_manifest(validation_config, manifest)
    validation_config = with_validation_output_dir(validation_config, output_dir)
    validation_config = with_validation_limit(validation_config, limit_samples)
    validation_config = with_validation_checkpoint(validation_config, checkpoint_path)
    validation_config = with_validation_lpips(validation_config, False if no_lpips else None)
    validation_config = with_validation_mlflow_disabled(validation_config, disable_mlflow)
    console.print(f"[bold]Validating EMA-VFI-small candidate:[/bold] {validation_config.candidate_id}")

    with _progress() as progress:
        samples_task = progress.add_task("Candidate samples", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                progress.update(samples_task, total=int(payload["total"]))
            elif event == "sample_advanced":
                progress.advance(samples_task)

        try:
            result = validate_candidate(validation_config, progress_callback=on_progress)
        except (MlflowLoggingError, ModelAdapterError, FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _print_candidate_validation_summary(result)


@amt_app.command("adapter-check")
def amt_adapter_check(
    config: Path = typer.Option(
        Path("configs/models/amt_s.yaml"),
        "--config",
        help="YAML config with AMT-S adapter parameters.",
    ),
    fail_on_blocked: bool = typer.Option(
        False,
        "--fail-on-blocked/--no-fail-on-blocked",
        help="Return a non-zero exit code when compatibility is blocked.",
    ),
) -> None:
    """Check AMT-S adapter prerequisites without running inference."""
    adapter_config = AMTAdapterConfig.from_mapping(_load_yaml_mapping(config))
    adapter = AMTAdapter(adapter_config)
    report = adapter.validate_environment()
    adapter.close()
    _print_adapter_environment_report("AMT-S adapter", report)
    if report.status == "failed" or (report.status == "blocked" and fail_on_blocked):
        raise typer.Exit(code=1)


@amt_app.command("infer-video")
def amt_infer_video(
    config: Path = typer.Option(
        Path("configs/inference/amt_s_2x.yaml"),
        "--config",
        help="YAML config with AMT-S local video inference parameters.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help="Override input video path.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        help="Override output video path.",
    ),
    limit_pairs: int | None = typer.Option(
        None,
        "--limit-pairs",
        min=1,
        help="Optional cap on interpolated neighboring frame pairs for smoke runs.",
    ),
    checkpoint_path: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Override AMT-S checkpoint path. Relative paths resolve under MODEL_WEIGHTS_ROOT.",
    ),
    codec: str | None = typer.Option(
        None,
        "--codec",
        help="Override FFmpeg/PyAV encoder name, for example libx264 or h264_nvenc.",
    ),
    disable_mlflow: bool = typer.Option(
        False,
        "--disable-mlflow",
        help="Skip MLflow logging for a tiny local smoke run.",
    ),
) -> None:
    """Run local 2x video inference with AMT-S."""
    inference_config = VideoInferenceConfig.from_mapping(
        _load_yaml_mapping(config),
        model_config_factory=AMTAdapterConfig.from_mapping,
    )
    inference_config = with_inference_config_path(inference_config, config)
    inference_config = with_inference_input(inference_config, input_path)
    inference_config = with_inference_output(inference_config, output_path)
    inference_config = with_inference_limit(inference_config, limit_pairs)
    inference_config = with_inference_checkpoint(inference_config, checkpoint_path)
    inference_config = with_inference_codec(inference_config, codec)
    inference_config = with_inference_mlflow_disabled(inference_config, disable_mlflow)
    console.print(f"[bold]Running AMT-S video inference:[/bold] {inference_config.input_path}")

    with _progress() as progress:
        pairs_task = progress.add_task("Frame pairs", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                total = int(payload.get("total") or 0)
                progress.update(pairs_task, total=total if total > 0 else None)
            elif event == "encoding_start":
                _print_inference_encoding_settings(payload)
            elif event == "pair_advanced":
                progress.advance(pairs_task)

        try:
            result = run_video_inference(
                inference_config,
                adapter_factory=lambda model_config, settings: AMTAdapter(model_config, settings=settings),
                progress_callback=on_progress,
                mlflow_mode="amt_video_inference",
            )
        except (MlflowLoggingError, ModelAdapterError, FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _print_video_inference_summary(result)


@amt_app.command("validate-candidate")
def amt_validate_candidate(
    config: Path = typer.Option(
        Path("configs/validation/amt_s_candidate.yaml"),
        "--config",
        help="YAML config with AMT-S candidate validation parameters.",
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="Override test manifest path. Use test_all.csv, not train/val manifests.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Override candidate validation output directory.",
    ),
    limit_samples: int | None = typer.Option(
        None,
        "--limit-samples",
        min=1,
        help="Optional sample cap for smoke runs.",
    ),
    checkpoint_path: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Override candidate checkpoint path. Relative paths resolve under MODEL_WEIGHTS_ROOT.",
    ),
    no_lpips: bool = typer.Option(
        False,
        "--no-lpips",
        help="Disable LPIPS for a fast CPU smoke run.",
    ),
    disable_mlflow: bool = typer.Option(
        False,
        "--disable-mlflow",
        help="Skip MLflow logging for a tiny local smoke run.",
    ),
) -> None:
    """Validate an AMT-S checkpoint candidate on a test manifest."""
    validation_config = CandidateValidationConfig.from_mapping(
        _load_yaml_mapping(config),
        model_config_factory=AMTAdapterConfig.from_mapping,
    )
    validation_config = with_validation_config_path(validation_config, config)
    validation_config = with_validation_manifest(validation_config, manifest)
    validation_config = with_validation_output_dir(validation_config, output_dir)
    validation_config = with_validation_limit(validation_config, limit_samples)
    validation_config = with_validation_checkpoint(validation_config, checkpoint_path)
    validation_config = with_validation_lpips(validation_config, False if no_lpips else None)
    validation_config = with_validation_mlflow_disabled(validation_config, disable_mlflow)
    console.print(f"[bold]Validating AMT-S candidate:[/bold] {validation_config.candidate_id}")

    with _progress() as progress:
        samples_task = progress.add_task("Candidate samples", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                progress.update(samples_task, total=int(payload["total"]))
            elif event == "sample_advanced":
                progress.advance(samples_task)

        try:
            result = validate_candidate_with_adapter(
                validation_config,
                adapter_factory=lambda model_config, settings: AMTAdapter(model_config, settings=settings),
                progress_callback=on_progress,
                model_adapter_label="AMTAdapter",
                mlflow_mode="amt_candidate_validation",
            )
        except (MlflowLoggingError, ModelAdapterError, FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _print_candidate_validation_summary(result)


@rife_app.command("adapter-check")
def rife_adapter_check(
    config: Path = typer.Option(
        Path("configs/models/practical_rife_v4_25.yaml"),
        "--config",
        help="YAML config with Practical-RIFE adapter parameters.",
    ),
    fail_on_blocked: bool = typer.Option(
        False,
        "--fail-on-blocked/--no-fail-on-blocked",
        help="Return a non-zero exit code when compatibility is blocked.",
    ),
) -> None:
    """Check Practical-RIFE adapter prerequisites without running inference."""
    adapter_config = PracticalRIFEAdapterConfig.from_mapping(_load_yaml_mapping(config))
    adapter = PracticalRIFEAdapter(adapter_config)
    report = adapter.validate_environment()
    adapter.close()
    _print_adapter_environment_report("Practical-RIFE adapter", report)
    if report.status == "failed" or (report.status == "blocked" and fail_on_blocked):
        raise typer.Exit(code=1)


@rife_app.command("infer-video")
def rife_infer_video(
    config: Path = typer.Option(
        Path("configs/inference/practical_rife_v4_25_2x.yaml"),
        "--config",
        help="YAML config with Practical-RIFE local video inference parameters.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help="Override input video path.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        help="Override output video path.",
    ),
    limit_pairs: int | None = typer.Option(
        None,
        "--limit-pairs",
        min=1,
        help="Optional cap on interpolated neighboring frame pairs for smoke runs.",
    ),
    checkpoint_path: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Override Practical-RIFE train_log checkpoint directory. Relative paths resolve under MODEL_WEIGHTS_ROOT.",
    ),
    codec: str | None = typer.Option(
        None,
        "--codec",
        help="Override FFmpeg/PyAV encoder name, for example libx264 or h264_nvenc.",
    ),
    disable_mlflow: bool = typer.Option(
        False,
        "--disable-mlflow",
        help="Skip MLflow logging for a tiny local smoke run.",
    ),
) -> None:
    """Run local 2x video inference with Practical-RIFE."""
    inference_config = VideoInferenceConfig.from_mapping(
        _load_yaml_mapping(config),
        model_config_factory=PracticalRIFEAdapterConfig.from_mapping,
    )
    inference_config = with_inference_config_path(inference_config, config)
    inference_config = with_inference_input(inference_config, input_path)
    inference_config = with_inference_output(inference_config, output_path)
    inference_config = with_inference_limit(inference_config, limit_pairs)
    inference_config = with_inference_checkpoint(inference_config, checkpoint_path)
    inference_config = with_inference_codec(inference_config, codec)
    inference_config = with_inference_mlflow_disabled(inference_config, disable_mlflow)
    console.print(f"[bold]Running Practical-RIFE video inference:[/bold] {inference_config.input_path}")

    with _progress() as progress:
        pairs_task = progress.add_task("Frame pairs", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                total = int(payload.get("total") or 0)
                progress.update(pairs_task, total=total if total > 0 else None)
            elif event == "encoding_start":
                _print_inference_encoding_settings(payload)
            elif event == "pair_advanced":
                progress.advance(pairs_task)

        try:
            result = run_video_inference(
                inference_config,
                adapter_factory=lambda model_config, settings: PracticalRIFEAdapter(model_config, settings=settings),
                progress_callback=on_progress,
                mlflow_mode="practical_rife_video_inference",
            )
        except (MlflowLoggingError, ModelAdapterError, FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _print_video_inference_summary(result)


@rife_app.command("validate-candidate")
def rife_validate_candidate(
    config: Path = typer.Option(
        Path("configs/validation/practical_rife_v4_25_candidate.yaml"),
        "--config",
        help="YAML config with Practical-RIFE candidate validation parameters.",
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="Override test manifest path. Use test_all.csv, not train/val manifests.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Override candidate validation output directory.",
    ),
    limit_samples: int | None = typer.Option(
        None,
        "--limit-samples",
        min=1,
        help="Optional sample cap for smoke runs.",
    ),
    checkpoint_path: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Override candidate train_log checkpoint directory. Relative paths resolve under MODEL_WEIGHTS_ROOT.",
    ),
    no_lpips: bool = typer.Option(
        False,
        "--no-lpips",
        help="Disable LPIPS for a fast CPU smoke run.",
    ),
    disable_mlflow: bool = typer.Option(
        False,
        "--disable-mlflow",
        help="Skip MLflow logging for a tiny local smoke run.",
    ),
) -> None:
    """Validate a Practical-RIFE checkpoint candidate on a test manifest."""
    validation_config = CandidateValidationConfig.from_mapping(
        _load_yaml_mapping(config),
        model_config_factory=PracticalRIFEAdapterConfig.from_mapping,
    )
    validation_config = with_validation_config_path(validation_config, config)
    validation_config = with_validation_manifest(validation_config, manifest)
    validation_config = with_validation_output_dir(validation_config, output_dir)
    validation_config = with_validation_limit(validation_config, limit_samples)
    validation_config = with_validation_checkpoint(validation_config, checkpoint_path)
    validation_config = with_validation_lpips(validation_config, False if no_lpips else None)
    validation_config = with_validation_mlflow_disabled(validation_config, disable_mlflow)
    console.print(f"[bold]Validating Practical-RIFE candidate:[/bold] {validation_config.candidate_id}")

    with _progress() as progress:
        samples_task = progress.add_task("Candidate samples", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                progress.update(samples_task, total=int(payload["total"]))
            elif event == "sample_advanced":
                progress.advance(samples_task)

        try:
            result = validate_candidate_with_adapter(
                validation_config,
                adapter_factory=lambda model_config, settings: PracticalRIFEAdapter(model_config, settings=settings),
                progress_callback=on_progress,
                model_adapter_label="PracticalRIFEAdapter",
                mlflow_mode="practical_rife_candidate_validation",
            )
        except (MlflowLoggingError, ModelAdapterError, FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _print_candidate_validation_summary(result)


@data_app.command("index-vimeo-triplets")
def index_vimeo_triplets(
    config: Path = typer.Option(
        Path("configs/data/index_vimeo_triplet.yaml"),
        "--config",
        help="YAML config with Vimeo source-indexing parameters.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Optional output CSV path. Defaults to the source dataset sequence_index.csv.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Optional maximum number of rows for smoke runs.",
    ),
) -> None:
    """Build a source-level sequence_index.csv for existing Vimeo triplets."""
    index_config = VimeoTripletIndexConfig.from_mapping(_load_yaml_mapping(config))
    index_config = with_vimeo_limit(index_config, limit)
    index_config = with_output_path(index_config, output)
    console.print(f"[bold]Indexing source dataset:[/bold] {index_config.source_dataset}")

    with _progress() as progress:
        records_task = progress.add_task("Vimeo records", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                progress.update(records_task, total=int(payload["total"]))
            elif event == "record_advanced":
                progress.advance(records_task)

        result = build_vimeo_triplet_index(index_config, progress_callback=on_progress)

    _print_index_summary(result)


@data_app.command("build-global-index")
def build_global_index(
    config: Path = typer.Option(
        Path("configs/data/global_index.yaml"),
        "--config",
        help="YAML config with global sequence-index parameters.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Optional output CSV path. Defaults to DATASET_ROOT/global_sequence_index.csv.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Optional maximum number of rows for smoke runs.",
    ),
    source_index: list[Path] | None = typer.Option(
        None,
        "--source-index",
        help="Explicit source sequence_index.csv path. Can be passed more than once.",
    ),
) -> None:
    """Combine source-level sequence indexes into one global index."""
    global_config = GlobalIndexConfig.from_mapping(_load_yaml_mapping(config))
    global_config = with_global_limit(global_config, limit)
    global_config = with_global_output_path(global_config, output)
    global_config = with_source_index_paths(global_config, source_index)
    console.print("[bold]Building global sequence index[/bold]")

    with _progress() as progress:
        records_task = progress.add_task("Global rows", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                progress.update(records_task, total=int(payload["total"]))
            elif event == "record_advanced":
                progress.advance(records_task)

        result = build_global_sequence_index(global_config, progress_callback=on_progress)

    _print_global_index_summary(result)


@data_app.command("build-dataset-version")
def build_dataset_version_command(
    config: Path = typer.Option(
        Path("configs/data/dataset_version.yaml"),
        "--config",
        help="YAML config with dataset-version parameters.",
    ),
    dataset_version_id: str | None = typer.Option(
        None,
        "--dataset-version-id",
        help="Override the dataset version id from the config.",
    ),
    source_index: Path | None = typer.Option(
        None,
        "--source-index",
        help="Override the global sequence index path.",
    ),
    output_root: Path | None = typer.Option(
        None,
        "--output-root",
        help="Override the dataset_versions output root.",
    ),
    limit_sequences: int | None = typer.Option(
        None,
        "--limit-sequences",
        min=1,
        help="Optional maximum number of sequence records for smoke runs.",
    ),
) -> None:
    """Build train/val/test triplet manifests for a dataset version."""
    version_config = DatasetVersionConfig.from_mapping(_load_yaml_mapping(config))
    version_config = with_dataset_version_id(version_config, dataset_version_id)
    version_config = with_dataset_source_index(version_config, source_index)
    version_config = with_dataset_output_root(version_config, output_root)
    version_config = with_dataset_limit_sequences(version_config, limit_sequences)
    console.print(f"[bold]Building dataset version:[/bold] {version_config.dataset_version_id}")

    with _progress() as progress:
        records_task = progress.add_task("Source records", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                progress.update(records_task, total=int(payload["total"]))
            elif event == "record_advanced":
                progress.advance(records_task)

        result = build_dataset_version(version_config, progress_callback=on_progress)

    _print_dataset_version_summary(result)


@data_app.command("inspect-triplet-manifest")
def inspect_triplet_manifest(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        help="Triplet manifest CSV to load through UniversalTripletDataset.",
    ),
    limit_samples: int = typer.Option(
        3,
        "--limit-samples",
        min=1,
        help="Maximum number of samples to inspect.",
    ),
) -> None:
    """Load a triplet manifest and print sample tensor shapes and metadata."""
    dataset = UniversalTripletDataset(manifest_path=manifest)
    _print_triplet_dataset_summary(dataset, limit_samples)


@data_app.command("preprocess-videos")
def preprocess_video_sources(
    config: Path = typer.Option(
        Path("configs/data/preprocess_anime.yaml"),
        "--config",
        help="YAML config with raw video preprocessing parameters.",
    ),
    limit_videos: int | None = typer.Option(
        None,
        "--limit-videos",
        min=1,
        help="Optional maximum number of input videos for smoke runs.",
    ),
    only_video: str | None = typer.Option(
        None,
        "--only-video",
        help="Process one video by filename or path relative to the raw input directory.",
    ),
    video_glob: str | None = typer.Option(
        None,
        "--video-glob",
        help="Process videos whose filename or relative path matches this glob.",
    ),
    max_duration_sec: float | None = typer.Option(
        None,
        "--max-duration-sec",
        min=0.001,
        help="Limit each video to this many seconds for debug/smoke runs.",
    ),
    max_frames: int | None = typer.Option(
        None,
        "--max-frames",
        min=1,
        help="Limit each video to this many frames for debug/smoke runs.",
    ),
    debug_progress: bool = typer.Option(
        False,
        "--debug-progress",
        help="Print timed per-video preprocessing diagnostics.",
    ),
) -> None:
    """Sample raw videos into PNG sequences and write a source sequence index."""
    preprocess_config = VideoPreprocessConfig.from_mapping(_load_yaml_mapping(config))
    preprocess_config = with_debug_overrides(
        preprocess_config,
        limit_videos=limit_videos,
        only_video=only_video,
        video_glob=video_glob,
        max_duration_sec=max_duration_sec,
        max_frames=max_frames,
    )
    console.print(f"[bold]Preprocessing videos:[/bold] {preprocess_config.raw_input_dir}")

    with _progress() as progress:
        videos_task = progress.add_task("Videos", total=None)
        sequences_task = progress.add_task("Current video", total=1, visible=False)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                progress.update(videos_task, total=int(payload["total"]))
            elif event == "video_start":
                description = f"Sequences: {payload['relative_video_path']}"
                progress.update(sequences_task, description=description, total=1, completed=0, visible=True)
                if debug_progress:
                    console.log(f"selected video: {payload['video_path']}")
            elif event == "debug_step_start" and debug_progress:
                console.log(f"{payload['step']} started")
            elif event == "metadata_done" and debug_progress:
                _print_metadata_debug(payload)
            elif event == "scene_detection_done" and debug_progress:
                console.log(
                    "scene detection finished "
                    f"elapsed={_format_seconds(payload['elapsed_sec'])} "
                    f"scenes={payload['scenes']} "
                    f"processed_frames={payload['processed_frames']} "
                    f"fallback={payload['fallback_used']}"
                )
                if payload["warning"]:
                    console.log(f"scene detection warning: {payload['warning']}")
            elif event == "candidate_sampling_done" and debug_progress:
                console.log(
                    "candidate sampling finished "
                    f"elapsed={_format_seconds(payload['elapsed_sec'])} "
                    f"candidates={payload['candidates']} "
                    f"unique_selected_frames={payload['unique_frame_indices']}"
                )
            elif event == "video_candidates":
                total = int(payload["candidates"])
                progress.update(sequences_task, total=max(total, 1), completed=0, visible=total > 0)
            elif event == "decode_done" and debug_progress:
                console.log(
                    "selected frame decoding finished "
                    f"elapsed={_format_seconds(payload['elapsed_sec'])} "
                    f"strategy={payload['strategy']} "
                    f"requested={payload['requested_count']} "
                    f"recovered={payload['recovered_count']} "
                    f"missing={payload['missing_count']}"
                )
            elif event == "write_done" and debug_progress:
                console.log(
                    "SSIM filtering and PNG writing finished "
                    f"elapsed={_format_seconds(payload['elapsed_sec'])} "
                    f"sequences={payload['sequences_written']} "
                    f"static_rejected={payload['static_triplets_rejected']}"
                )
            elif event == "sequence_advanced":
                progress.advance(sequences_task)
            elif event == "video_done":
                console.print(
                    "[green]video done[/green] "
                    f"status={payload['status']} "
                    f"sequences={payload['sequences_written']} "
                    f"static_rejected={payload['static_triplets_rejected']}"
                )
            elif event == "video_failed":
                console.print(f"[yellow]video failed[/yellow] {payload['error']}")
            elif event == "video_summary" and debug_progress:
                console.log(
                    "per-video summary "
                    f"elapsed={_format_seconds(payload['elapsed_sec'])} "
                    f"status={payload['status']} "
                    f"sequences={payload['sequences_written']} "
                    f"static_rejected={payload['static_triplets_rejected']}"
                )
            elif event == "video_advanced":
                progress.advance(videos_task)

        result = preprocess_videos(preprocess_config, progress_callback=on_progress)

    _print_preprocess_summary(result)


@baseline_app.command("evaluate")
def evaluate_baseline_command(
    config: Path = typer.Option(
        Path("configs/baselines/baseline_eval.yaml"),
        "--config",
        help="YAML config with baseline evaluation parameters.",
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="Override the triplet manifest path, usually a test_all.csv.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Override the baseline output directory.",
    ),
    limit_samples: int | None = typer.Option(
        None,
        "--limit-samples",
        min=1,
        help="Optional sample cap for smoke runs.",
    ),
    baseline: list[str] | None = typer.Option(
        None,
        "--baseline",
        help="Baseline to run. Repeat for multiple values: duplicate_left, blend, farneback.",
    ),
    no_lpips: bool = typer.Option(
        False,
        "--no-lpips",
        help="Disable LPIPS for a fast CPU smoke run.",
    ),
    no_predictions: bool = typer.Option(
        False,
        "--no-predictions",
        help="Do not write sample prediction PNGs.",
    ),
    disable_mlflow: bool = typer.Option(
        False,
        "--disable-mlflow",
        help="Skip MLflow logging for a tiny local smoke run.",
    ),
) -> None:
    """Evaluate simple interpolation baselines on a triplet manifest."""
    baseline_config = BaselineEvaluationConfig.from_mapping(_load_yaml_mapping(config))
    baseline_config = with_baseline_config_path(baseline_config, config)
    baseline_config = with_baseline_manifest(baseline_config, manifest)
    baseline_config = with_baseline_output_dir(baseline_config, output_dir)
    baseline_config = with_baseline_limit(baseline_config, limit_samples)
    baseline_config = with_baseline_names(baseline_config, baseline)
    baseline_config = with_baseline_lpips(baseline_config, False if no_lpips else None)
    baseline_config = with_baseline_save_predictions(baseline_config, False if no_predictions else None)
    baseline_config = with_baseline_mlflow_disabled(baseline_config, disable_mlflow)
    console.print(f"[bold]Evaluating baselines:[/bold] {', '.join(baseline_config.baselines)}")

    with _progress() as progress:
        predictions_task = progress.add_task("Baseline predictions", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                progress.update(predictions_task, total=int(payload["total"]))
            elif event == "prediction_advanced":
                progress.advance(predictions_task)

        try:
            result = evaluate_baselines(baseline_config, progress_callback=on_progress)
        except MlflowLoggingError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _print_baseline_summary(result)


@baseline_app.command("infer-video")
def baseline_infer_video(
    config: Path = typer.Option(
        Path("configs/inference/baseline_2x.yaml"),
        "--config",
        help="YAML config with baseline local video inference parameters.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help="Override input video path.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        help="Override output video path.",
    ),
    baseline_name: str | None = typer.Option(
        None,
        "--baseline",
        help="Baseline method to run: duplicate_left, blend, or farneback.",
    ),
    limit_pairs: int | None = typer.Option(
        None,
        "--limit-pairs",
        min=1,
        help="Optional cap on interpolated neighboring frame pairs for smoke runs.",
    ),
    codec: str | None = typer.Option(
        None,
        "--codec",
        help="Override FFmpeg/PyAV encoder name, for example libx264 or h264_nvenc.",
    ),
    disable_mlflow: bool = typer.Option(
        False,
        "--disable-mlflow",
        help="Skip MLflow logging for a tiny local smoke run.",
    ),
) -> None:
    """Run local 2x video inference with a non-neural baseline."""
    inference_config = VideoInferenceConfig.from_mapping(
        _load_yaml_mapping(config),
        model_config_factory=BaselineAdapterConfig.from_mapping,
    )
    inference_config = with_inference_config_path(inference_config, config)
    inference_config = with_inference_input(inference_config, input_path)
    inference_config = with_inference_output(inference_config, output_path)
    inference_config = with_inference_limit(inference_config, limit_pairs)
    inference_config = with_inference_codec(inference_config, codec)
    inference_config = with_inference_mlflow_disabled(inference_config, disable_mlflow)
    inference_config = replace(
        inference_config,
        model=with_baseline_adapter_name(inference_config.model, baseline_name),
    )
    console.print(
        "[bold]Running baseline video inference:[/bold] "
        f"{inference_config.model.baseline_name} on {inference_config.input_path}"
    )

    with _progress() as progress:
        pairs_task = progress.add_task("Frame pairs", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                total = int(payload.get("total") or 0)
                progress.update(pairs_task, total=total if total > 0 else None)
            elif event == "encoding_start":
                _print_inference_encoding_settings(payload)
            elif event == "pair_advanced":
                progress.advance(pairs_task)

        try:
            result = run_video_inference(
                inference_config,
                adapter_factory=lambda model_config, _settings: BaselineAdapter(model_config),
                progress_callback=on_progress,
                mlflow_mode="baseline_video_inference",
            )
        except (MlflowLoggingError, ModelAdapterError, FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _print_video_inference_summary(result)


@mlflow_app.command("smoke-log")
def mlflow_smoke_log(
    experiment_name: str = typer.Option(
        "stage1-smoke",
        "--experiment-name",
        help="MLflow experiment name to create/use for the smoke run.",
    ),
    run_name: str = typer.Option(
        "stage1-mlflow-smoke",
        "--run-name",
        help="MLflow run name.",
    ),
    artifact_path: Path = typer.Option(
        Path("/tmp/stage1_mlflow_smoke.txt"),
        "--artifact-path",
        help="Small local artifact file to create and log.",
    ),
) -> None:
    """Log a minimal MLflow run to the configured tracking URI."""
    try:
        result = run_mlflow_smoke(
            MlflowRunConfig(experiment_name=experiment_name, run_name=run_name),
            artifact_path=artifact_path,
        )
    except MlflowLoggingError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    _print_mlflow_smoke_summary(result)


def _inference_targets() -> list[_InferenceTarget]:
    # Practical-RIFE goes before EMA because both upstream repos use a top-level `model` package name.
    return [
        _InferenceTarget(
            name="practical_rife_v4_25",
            label="Practical-RIFE v4.25",
            config_path=Path("configs/inference/practical_rife_v4_25_2x.yaml"),
            output_group=Path("practical_rife_v4_25"),
            model_config_factory=PracticalRIFEAdapterConfig.from_mapping,
            adapter_factory=lambda model_config, settings: PracticalRIFEAdapter(model_config, settings=settings),
            mlflow_mode="practical_rife_video_inference",
        ),
        _InferenceTarget(
            name="amt_s",
            label="AMT-S",
            config_path=Path("configs/inference/amt_s_2x.yaml"),
            output_group=Path("amt_s"),
            model_config_factory=AMTAdapterConfig.from_mapping,
            adapter_factory=lambda model_config, settings: AMTAdapter(model_config, settings=settings),
            mlflow_mode="amt_video_inference",
        ),
        _InferenceTarget(
            name="ema_vfi_small",
            label="EMA-VFI-small",
            config_path=Path("configs/inference/ema_vfi_small_2x.yaml"),
            output_group=Path("ema_vfi_small"),
            model_config_factory=EMAVFIAdapterConfig.from_mapping,
            adapter_factory=lambda model_config, settings: EMAVFIAdapter(model_config, settings=settings),
            mlflow_mode="ema_video_inference",
        ),
        *[
            _InferenceTarget(
                name=f"baseline_{baseline_name}",
                label=f"baseline {baseline_name}",
                config_path=Path("configs/inference/baseline_2x.yaml"),
                output_group=Path("baselines") / baseline_name,
                model_config_factory=BaselineAdapterConfig.from_mapping,
                adapter_factory=lambda model_config, _settings: BaselineAdapter(model_config),
                mlflow_mode="baseline_video_inference",
                baseline_name=baseline_name,
            )
            for baseline_name in ("duplicate_left", "blend", "farneback")
        ],
    ]


def _selected_inference_targets(
    targets: list[_InferenceTarget],
    target_selection: list[str] | None,
) -> list[_InferenceTarget]:
    selected_names = resolve_batch_target_names(
        [target.name for target in targets],
        target_selection,
        aliases=BATCH_TARGET_ALIASES,
    )
    by_name = {target.name: target for target in targets}
    return [by_name[name] for name in selected_names]


def _load_batch_target_config(
    target: _InferenceTarget,
    *,
    limit_pairs: int | None,
    codec: str | None,
    disable_mlflow: bool,
) -> VideoInferenceConfig:
    inference_config = VideoInferenceConfig.from_mapping(
        _load_yaml_mapping(target.config_path),
        model_config_factory=target.model_config_factory,
    )
    inference_config = with_inference_config_path(inference_config, target.config_path)
    inference_config = with_inference_limit(inference_config, limit_pairs)
    inference_config = with_inference_codec(inference_config, codec)
    inference_config = with_inference_mlflow_disabled(inference_config, disable_mlflow)
    if target.baseline_name is not None:
        inference_config = replace(
            inference_config,
            model=with_baseline_adapter_name(inference_config.model, target.baseline_name),
        )
    return inference_config


def _batch_video_config(
    target_config: VideoInferenceConfig,
    target: _InferenceTarget,
    video: BatchInferenceVideo,
    output_root: Path,
) -> VideoInferenceConfig:
    output_extension = target_config.output_path.suffix or ".mp4"
    output_path = batch_output_path(
        output_root=output_root,
        output_group=target.output_group,
        relative_input_path=video.relative_path,
        output_extension=output_extension,
    )
    return replace(
        target_config,
        input_path=video.input_path,
        output_path=output_path,
        mlflow=replace(
            target_config.mlflow,
            run_name=batch_run_name(target_name=target.name, relative_input_path=video.relative_path),
        ),
    )


def _write_batch_measurement_files(
    records: list[_BatchInferenceRecord],
    targets: list[_InferenceTarget],
    output_root: Path,
) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        target_records = [record for record in records if record.target_name == target.name]
        if not target_records:
            continue
        path = batch_measurements_path(output_root=output_root, output_group=target.output_group)
        write_batch_measurements_csv(path, [_batch_measurement_row(record) for record in target_records])
        paths.append(path)
    return paths


def _batch_measurement_row(record: _BatchInferenceRecord) -> dict[str, object]:
    return {
        "target_name": record.target_name,
        "input_video": record.input_video,
        "relative_input_video": record.relative_input_video,
        "output_video": record.output_video,
        "status": record.status,
        "pairs_processed": record.pairs_processed,
        "frames_written": record.frames_written,
        "input_fps": _optional_float(record.input_fps),
        "output_fps": _optional_float(record.output_fps),
        "model_inference_elapsed_sec": _optional_float(record.model_inference_elapsed_sec),
        "total_elapsed_sec": _optional_float(record.total_elapsed_sec),
        "model_pairs_per_sec": _optional_float(record.model_pairs_per_sec),
        "total_pairs_per_sec": _optional_float(record.total_pairs_per_sec),
        "audio_streams_available": record.audio_streams_available,
        "audio_streams_preserved": record.audio_streams_preserved,
        "mlflow_run_id": record.mlflow_run_id,
        "error": record.error,
    }


def _optional_float(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.8f}"


def _print_index_summary(result: SourceIndexResult) -> None:
    table = Table(title="Source Index Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("source dataset", result.source_dataset)
    table.add_row("records read", str(result.records_read))
    table.add_row("records written", str(result.rows_written))
    table.add_row("invalid/skipped records", str(result.invalid_records))
    table.add_row("output index path", str(result.index_path))
    console.print(table)


def _print_global_index_summary(result: GlobalIndexResult) -> None:
    table = Table(title="Global Index Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("source indexes read", str(result.source_indexes_read))
    table.add_row("records read", str(result.records_read))
    table.add_row("records written", str(result.rows_written))
    table.add_row("filtered records", str(result.filtered_records))
    table.add_row("output index path", str(result.index_path))
    console.print(table)


def _print_dataset_version_summary(result: DatasetVersionResult) -> None:
    table = Table(title="Dataset Version Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("output directory", str(result.output_dir))
    table.add_row("source records read", str(result.source_records_read))
    table.add_row("samples generated", str(result.samples_generated))
    table.add_row("train samples", str(result.train_samples))
    table.add_row("val samples", str(result.val_samples))
    table.add_row("test samples", str(result.test_samples))
    table.add_row("train source videos", str(result.train_source_videos))
    table.add_row("val source videos", str(result.val_source_videos))
    table.add_row("test source videos", str(result.test_source_videos))
    table.add_row("train manifest", str(result.train_manifest_path))
    table.add_row("val manifest", str(result.val_manifest_path))
    table.add_row("test manifest", str(result.test_manifest_path))
    table.add_row("dataset_config.yaml", str(result.dataset_config_path))
    console.print(table)


def _print_triplet_dataset_summary(dataset: UniversalTripletDataset, limit_samples: int) -> None:
    table = Table(title="Triplet Dataset Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("manifest", str(dataset.manifest_path))
    table.add_row("dataset root", str(dataset.dataset_root))
    table.add_row("samples", str(len(dataset)))
    console.print(table)

    sample_table = Table(title="Sample Shapes")
    sample_table.add_column("sample_id")
    sample_table.add_column("left")
    sample_table.add_column("middle")
    sample_table.add_column("right")
    sample_table.add_column("source")
    for index in range(min(len(dataset), limit_samples)):
        sample = dataset[index]
        metadata = sample["metadata"]
        sample_table.add_row(
            metadata["sample_id"],
            "x".join(str(value) for value in sample["left"].shape),
            "x".join(str(value) for value in sample["middle"].shape),
            "x".join(str(value) for value in sample["right"].shape),
            f"{metadata['source_group']}/{metadata['source_dataset']}",
        )
    console.print(sample_table)


def _print_baseline_summary(result: BaselineEvaluationResult) -> None:
    table = Table(title="Baseline Evaluation Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("output directory", str(result.output_dir))
    table.add_row("samples evaluated", str(result.samples_evaluated))
    table.add_row("baseline predictions", str(result.predictions_evaluated))
    table.add_row("baselines", ", ".join(result.baselines))
    table.add_row("sample predictions written", str(result.sample_predictions_written))
    table.add_row("metrics.csv", str(result.metrics_csv_path))
    table.add_row("metrics_summary.csv", str(result.summary_csv_path))
    table.add_row("mlflow run id", result.mlflow_run_id or "not logged")
    console.print(table)

    summary_table = Table(title="Aggregate Metrics")
    summary_table.add_column("prediction")
    summary_table.add_column("source_group")
    summary_table.add_column("source_dataset")
    summary_table.add_column("count")
    summary_table.add_column("PSNR")
    summary_table.add_column("SSIM")
    summary_table.add_column("LPIPS")
    for row in result.summary_rows:
        if row["source_group"] == "__all__" or row["source_dataset"] == "__all__":
            summary_table.add_row(
                row["prediction_name"],
                row["source_group"],
                row["source_dataset"],
                row["count"],
                row["psnr_mean"],
                row["ssim_mean"],
                row["lpips_mean"],
            )
    console.print(summary_table)


def _print_mlflow_smoke_summary(result: MlflowSmokeResult) -> None:
    table = Table(title="MLflow Smoke Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("tracking URI", result.tracking_uri)
    table.add_row("experiment", result.experiment_name)
    table.add_row("run id", result.run_id)
    console.print(table)


def _print_batch_inference_summary(records: list[_BatchInferenceRecord], measurement_paths: list[Path]) -> None:
    ok_count = sum(1 for record in records if record.status == "ok")
    failed_count = len(records) - ok_count
    table = Table(title="Batch Inference Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("jobs total", str(len(records)))
    table.add_row("jobs completed", str(ok_count))
    table.add_row("jobs failed", str(failed_count))
    console.print(table)

    detail = Table(title="Batch Inference Outputs")
    detail.add_column("target")
    detail.add_column("input")
    detail.add_column("status")
    detail.add_column("output")
    detail.add_column("pairs")
    detail.add_column("elapsed")
    detail.add_column("mlflow")
    detail.add_column("error")
    for record in records:
        detail.add_row(
            record.target_name,
            str(record.input_video),
            record.status,
            "" if record.output_video is None else str(record.output_video),
            "" if record.pairs_processed is None else str(record.pairs_processed),
            "" if record.total_elapsed_sec is None else f"{record.total_elapsed_sec:.3f}s",
            record.mlflow_run_id or "",
            (record.error or "")[:120],
        )
    console.print(detail)

    if measurement_paths:
        measurements = Table(title="Batch Measurement CSVs")
        measurements.add_column("Path")
        for path in measurement_paths:
            measurements.add_row(str(path))
        console.print(measurements)


def _print_adapter_environment_report(title: str, report: AdapterEnvironmentReport) -> None:
    table = Table(title=f"{title}: {report.status}")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in report.checks:
        table.add_row(check.name, check.status, check.detail)
    console.print(table)


def _print_video_inference_summary(result: VideoInferenceResult) -> None:
    table = Table(title="Video Inference Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("input video", str(result.input_path))
    table.add_row("output video", str(result.output_path))
    table.add_row("input fps", f"{result.input_fps:.4f}")
    table.add_row("output fps", f"{result.output_fps:.4f}")
    table.add_row("codec", result.codec)
    table.add_row("container", result.container or "inferred")
    table.add_row("pixel format", result.pix_fmt)
    table.add_row("frame format", result.frame_format)
    table.add_row("audio streams", f"{result.audio_streams_preserved}/{result.audio_streams_available} preserved")
    table.add_row("pairs processed", str(result.pairs_processed))
    table.add_row("frames written", str(result.frames_written))
    table.add_row("model inference time", f"{result.model_inference_elapsed_sec:.3f}s")
    table.add_row("total elapsed time", f"{result.total_elapsed_sec:.3f}s")
    table.add_row("model pairs/sec", f"{result.model_pairs_per_sec:.4f}")
    table.add_row("total pairs/sec", f"{result.total_pairs_per_sec:.4f}")
    table.add_row("mlflow run id", result.mlflow_run_id or "not logged")
    console.print(table)


def _print_inference_encoding_settings(payload: dict[str, object]) -> None:
    table = Table(title="PyAV Encoding Settings")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("output path", str(payload["output_path"]))
    table.add_row("container", str(payload["container"]))
    table.add_row("codec", str(payload["codec"]))
    table.add_row("output fps", f"{float(payload['output_fps']):.4f}")
    table.add_row("frame size", f"{payload['width']}x{payload['height']}")
    table.add_row("pixel format", str(payload["pix_fmt"]))
    table.add_row("frame format", str(payload["frame_format"]))
    table.add_row("encoder options", str(payload["encoder_options"]))
    table.add_row(
        "audio streams",
        f"{payload['audio_streams_to_preserve']}/{payload['audio_streams_available']} to preserve",
    )
    console.print(table)


def _print_ema_training_summary(result: EMATrainingResult) -> None:
    table = Table(title="EMA Training Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("output directory", str(result.output_dir))
    table.add_row("steps completed", str(result.steps_completed))
    table.add_row("best psnr", "" if result.best_psnr is None else f"{result.best_psnr:.8f}")
    table.add_row("best checkpoint", str(result.best_checkpoint_path or "not written"))
    table.add_row("last checkpoint", str(result.last_checkpoint_path or "not written"))
    table.add_row("mlflow run id", result.mlflow_run_id or "not logged")
    console.print(table)


def _print_candidate_validation_summary(result: CandidateValidationResult) -> None:
    table = Table(title="Candidate Validation Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("output directory", str(result.output_dir))
    table.add_row("samples evaluated", str(result.samples_evaluated))
    table.add_row("sample predictions written", str(result.sample_predictions_written))
    table.add_row("decision", "approved" if result.approved else "rejected")
    table.add_row("metrics.csv", str(result.metrics_csv_path))
    table.add_row("metrics_summary.csv", str(result.summary_csv_path))
    table.add_row("report", str(result.report_path))
    table.add_row("mlflow run id", result.mlflow_run_id or "not logged")
    console.print(table)
    if result.decision_reasons:
        console.print("[bold]Decision reasons:[/bold]")
        for reason in result.decision_reasons:
            console.print(f"- {reason}")


def _print_preprocess_summary(result: PreprocessResult) -> None:
    table = Table(title="Preprocessing Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("videos discovered", str(result.videos_discovered))
    table.add_row("videos processed", str(result.videos_processed))
    table.add_row("videos skipped/failed", f"{result.videos_skipped}/{result.videos_failed}")
    table.add_row("scenes detected", str(result.scenes_detected))
    table.add_row("sequences written", str(result.sequences_written))
    table.add_row("static triplets rejected", str(result.static_triplets_rejected))
    table.add_row("output directory", str(result.output_dir))
    table.add_row("sequence_index.csv", str(result.index_path))
    console.print(table)

    if result.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in result.warnings[:5]:
            console.print(f"- {warning}")
        if len(result.warnings) > 5:
            console.print(f"- ... {len(result.warnings) - 5} more warning(s)")


def _print_metadata_debug(payload: dict[str, object]) -> None:
    table = Table(title="Video Metadata")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("elapsed", _format_seconds(payload["elapsed_sec"]))
    table.add_row("container format", str(payload["container_format"]))
    table.add_row("video codec", str(payload["video_codec"]))
    table.add_row("audio codecs", str(payload["audio_codecs"]))
    table.add_row("width x height", f"{payload['width']} x {payload['height']}")
    table.add_row("fps", str(payload["fps"]))
    table.add_row("average_rate", str(payload["average_rate"]))
    table.add_row("stream.frames", str(payload["stream_frames"]))
    table.add_row("stream duration sec", str(payload["stream_duration_sec"]))
    table.add_row("container duration sec", str(payload["container_duration_sec"]))
    table.add_row("frame count used", str(payload["frame_count"]))
    table.add_row("source frame count", str(payload["source_frame_count"]))
    table.add_row("frame count source", str(payload["frame_count_source"]))
    table.add_row("frame count exact", str(payload["frame_count_exact"]))
    table.add_row("frame limit applied", str(payload["frame_limit_applied"]))
    console.print(table)


def _format_seconds(value: object) -> str:
    return f"{float(value):.3f}s"


app.add_typer(data_app, name="data")
app.add_typer(baseline_app, name="baseline")
app.add_typer(mlflow_app, name="mlflow")
app.add_typer(ema_app, name="ema")
app.add_typer(amt_app, name="amt")
app.add_typer(rife_app, name="rife")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
