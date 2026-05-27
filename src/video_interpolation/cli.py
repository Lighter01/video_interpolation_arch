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
from .adapters.base import AdapterEnvironmentReport, ModelAdapterError
from .adapters.ema_vfi import EMAVFIAdapter, EMAVFIAdapterConfig
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
    with_inference_checkpoint,
    with_inference_config_path,
    with_inference_input,
    with_inference_limit,
    with_inference_mlflow_disabled,
    with_inference_output,
)
from .mlflow import MlflowLoggingError, MlflowRunConfig, MlflowSmokeResult, run_mlflow_smoke
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
console = Console()


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


def _print_preflight_report(report: PreflightReport) -> None:
    table = Table(title=f"EMA-VFI-small preflight: {report.status}")
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
    inference_config = with_inference_mlflow_disabled(inference_config, disable_mlflow)
    console.print(f"[bold]Running EMA-VFI-small video inference:[/bold] {inference_config.input_path}")

    with _progress() as progress:
        pairs_task = progress.add_task("Frame pairs", total=None)

        def on_progress(event: str, payload: dict[str, object]) -> None:
            if event == "start":
                total = int(payload.get("total") or 0)
                progress.update(pairs_task, total=total if total > 0 else None)
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


def _print_adapter_environment_report(title: str, report: AdapterEnvironmentReport) -> None:
    table = Table(title=f"{title}: {report.status}")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in report.checks:
        table.add_row(check.name, check.status, check.detail)
    console.print(table)


def _print_video_inference_summary(result: VideoInferenceResult) -> None:
    table = Table(title="EMA Video Inference Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("input video", str(result.input_path))
    table.add_row("output video", str(result.output_path))
    table.add_row("input fps", f"{result.input_fps:.4f}")
    table.add_row("output fps", f"{result.output_fps:.4f}")
    table.add_row("pairs processed", str(result.pairs_processed))
    table.add_row("frames written", str(result.frames_written))
    table.add_row("mlflow run id", result.mlflow_run_id or "not logged")
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
