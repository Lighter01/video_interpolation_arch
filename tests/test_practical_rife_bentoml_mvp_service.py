from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pytest
from bentoml.exceptions import BadInput, InternalServerError


def test_mvp_request_validation_accepts_defaults_and_creates_output_directory(tmp_path: Path) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "nested" / "output.mp4"
    input_path.write_bytes(b"input")

    request = module.validate_interpolate_video_request(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "interpolation_factor": 2,
        }
    )

    assert request.input_path == input_path
    assert request.output_path == output_path
    assert request.interpolation_factor == 2
    assert request.scale == 1.0
    assert request.output_playback_mode.value == "real_time"
    assert request.quality_evaluation_enabled
    assert request.quality_sample_count == 16
    assert output_path.parent.is_dir()


@pytest.mark.parametrize(
    ("payload_overrides", "message"),
    [
        ({"input_path": "relative.mp4"}, "absolute"),
        ({"input_path": None}, "input_path"),
    ],
)
def test_mvp_request_validation_rejects_missing_or_invalid_input_path(
    tmp_path: Path,
    payload_overrides: dict[str, object],
    message: str,
) -> None:
    module = _load_service_module()
    output_path = tmp_path / "output.mp4"
    payload = {
        "input_path": str(tmp_path / "missing.mp4"),
        "output_path": str(output_path),
        "interpolation_factor": 2,
    }
    payload.update(payload_overrides)

    with pytest.raises(BadInput, match=message):
        module.validate_interpolate_video_request(payload)


def test_mvp_request_validation_rejects_missing_input_path(tmp_path: Path) -> None:
    module = _load_service_module()

    with pytest.raises(BadInput, match="input_path"):
        module.validate_interpolate_video_request(
            {
                "output_path": str(tmp_path / "output.mp4"),
                "interpolation_factor": 2,
            }
        )


def test_mvp_request_validation_rejects_nonexistent_input_file(tmp_path: Path) -> None:
    module = _load_service_module()

    with pytest.raises(BadInput, match="existing video file"):
        module.validate_interpolate_video_request(
            {
                "input_path": str(tmp_path / "missing.mp4"),
                "output_path": str(tmp_path / "output.mp4"),
                "interpolation_factor": 2,
            }
        )


@pytest.mark.parametrize(
    ("payload_overrides", "message"),
    [
        ({"output_path": "relative.mp4"}, "absolute"),
        ({"output_path": None}, "output_path"),
    ],
)
def test_mvp_request_validation_rejects_missing_or_invalid_output_path(
    tmp_path: Path,
    payload_overrides: dict[str, object],
    message: str,
) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")
    payload = {
        "input_path": str(input_path),
        "output_path": str(tmp_path / "output.mp4"),
        "interpolation_factor": 2,
    }
    payload.update(payload_overrides)

    with pytest.raises(BadInput, match=message):
        module.validate_interpolate_video_request(payload)


def test_mvp_request_validation_rejects_missing_output_path(tmp_path: Path) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")

    with pytest.raises(BadInput, match="output_path"):
        module.validate_interpolate_video_request(
            {
                "input_path": str(input_path),
                "interpolation_factor": 2,
            }
        )


def test_mvp_request_validation_rejects_output_path_that_is_directory(tmp_path: Path) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output_dir"
    input_path.write_bytes(b"input")
    output_path.mkdir()

    with pytest.raises(BadInput, match="directory"):
        module.validate_interpolate_video_request(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "interpolation_factor": 2,
            }
        )


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_mvp_request_validation_accepts_allowed_factors(tmp_path: Path, factor: int) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")

    request = module.validate_interpolate_video_request(
        {
            "input_path": str(input_path),
            "output_path": str(tmp_path / f"output_{factor}.mp4"),
            "interpolation_factor": factor,
        }
    )

    assert request.interpolation_factor == factor


@pytest.mark.parametrize("factor", [1, 5, True, "2"])
def test_mvp_request_validation_rejects_invalid_factors(tmp_path: Path, factor: object) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")

    with pytest.raises(BadInput, match="interpolation_factor|integer"):
        module.validate_interpolate_video_request(
            {
                "input_path": str(input_path),
                "output_path": str(tmp_path / "output.mp4"),
                "interpolation_factor": factor,
            }
        )


@pytest.mark.parametrize("scale", [0.75, True, "1.0"])
def test_mvp_request_validation_rejects_invalid_scale(tmp_path: Path, scale: object) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")

    with pytest.raises(BadInput, match="scale|number"):
        module.validate_interpolate_video_request(
            {
                "input_path": str(input_path),
                "output_path": str(tmp_path / "output.mp4"),
                "interpolation_factor": 2,
                "scale": scale,
            }
        )


def test_mvp_request_validation_rejects_invalid_output_playback_mode(tmp_path: Path) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")

    with pytest.raises(BadInput, match="output_playback_mode"):
        module.validate_interpolate_video_request(
            {
                "input_path": str(input_path),
                "output_path": str(tmp_path / "output.mp4"),
                "interpolation_factor": 2,
                "output_playback_mode": "cinema",
            }
        )


@pytest.mark.parametrize("sample_count", [-1, True, "16"])
def test_mvp_request_validation_rejects_invalid_quality_sample_count(
    tmp_path: Path,
    sample_count: object,
) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")

    with pytest.raises(BadInput, match="quality_sample_count|integer"):
        module.validate_interpolate_video_request(
            {
                "input_path": str(input_path),
                "output_path": str(tmp_path / "output.mp4"),
                "interpolation_factor": 2,
                "quality_sample_count": sample_count,
            }
        )


def test_mvp_service_contract_runs_fake_runner_and_formats_worker_response(tmp_path: Path) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "out" / "output.mp4"
    input_path.write_bytes(b"input")
    runner = _FakeRunner(write_output=True)

    response = module.run_interpolate_video_request(
        runner,
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "interpolation_factor": 3,
            "scale": 0.5,
            "output_playback_mode": "slow_motion",
            "quality_evaluation_enabled": False,
            "quality_sample_count": 4,
        },
    )

    assert response["status"] == "completed"
    assert response["output_path"] == str(output_path)
    assert response["interpolation_factor"] == 3
    assert response["duration_seconds"] >= 0.0
    assert response["psnr_mean"] == 31.42
    assert response["ssim_mean"] == 0.948
    assert "device" not in response
    assert "backend" not in response
    assert "runtime_backend" not in response
    assert "quality_triplets_written" not in response
    assert "quality_triplet_output_dir" not in response
    assert runner.calls == [
        {
            "input_path": input_path,
            "output_path": output_path,
            "interpolation_factor": 3,
            "scale": 0.5,
            "output_playback_mode": "slow_motion",
            "enable_quality_evaluation": False,
            "quality_sample_count": 4,
        }
    ]


def test_mvp_service_contract_rejects_missing_runner_output(tmp_path: Path) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")

    with pytest.raises(InternalServerError, match="did not create"):
        module.run_interpolate_video_request(
            _FakeRunner(write_output=False),
            {
                "input_path": str(input_path),
                "output_path": str(tmp_path / "missing.mp4"),
                "interpolation_factor": 2,
            },
        )


def test_mvp_service_contract_rejects_empty_runner_output(tmp_path: Path) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "empty.mp4"
    input_path.write_bytes(b"input")

    with pytest.raises(InternalServerError, match="empty"):
        module.run_interpolate_video_request(
            _FakeRunner(write_output=True, output_bytes=b""),
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "interpolation_factor": 2,
            },
        )


def test_mvp_service_config_defaults_to_cuda_nvenc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RIFE_DEVICE", raising=False)
    monkeypatch.delenv("RIFE_CODEC", raising=False)

    module = _load_service_module()

    assert module.SERVICE_CONFIG.backend.value == "torch"
    assert module.SERVICE_CONFIG.device == "cuda"
    assert module.SERVICE_CONFIG.codec == "h264_nvenc"


def test_mvp_service_config_reads_codec_and_device_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIFE_DEVICE", "cpu")
    monkeypatch.setenv("RIFE_CODEC", "libx264")

    module = _load_service_module()

    assert module.SERVICE_CONFIG.backend.value == "torch"
    assert module.SERVICE_CONFIG.device == "cpu"
    assert module.SERVICE_CONFIG.codec == "libx264"


@pytest.mark.parametrize("env_name", ["RIFE_DEVICE", "RIFE_CODEC"])
def test_mvp_service_config_rejects_empty_env_values(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
) -> None:
    monkeypatch.setenv(env_name, "  ")

    with pytest.raises(ValueError, match=env_name):
        _load_service_module()


def test_mvp_bentoml_service_exposes_interpolate_video_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RIFE_DEVICE", raising=False)
    monkeypatch.delenv("RIFE_CODEC", raising=False)
    module = _load_service_module()

    assert "interpolate_video" in module.PracticalRIFEInterpolationService.apis
    api = module.PracticalRIFEInterpolationService.apis["interpolate_video"]
    assert api.route == "/interpolate_video"


def test_mvp_docker_and_compose_files_define_gpu_shared_volume_contract() -> None:
    dockerfile = Path("services/practical_rife_bentoml/Dockerfile")
    compose_file = Path("services/practical_rife_bentoml/docker-compose.gpu.example.yml")

    docker_text = dockerfile.read_text(encoding="utf-8")
    compose_text = compose_file.read_text(encoding="utf-8")

    assert "bentoml" in docker_text
    assert "services/practical_rife_bentoml/service.py:PracticalRIFEInterpolationService" in docker_text
    assert "--host" in docker_text
    assert "0.0.0.0" in docker_text
    assert "--port" in docker_text
    assert "3000" in docker_text
    assert "PYTHONPATH=/app/src" in docker_text
    assert "RIFE_CODEC=h264_nvenc" in docker_text
    assert "NVIDIA_VISIBLE_DEVICES=all" in docker_text
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" in docker_text
    assert "COPY src ./src" in docker_text
    assert "COPY configs ./configs" in docker_text
    assert "COPY services/practical_rife_bentoml ./services/practical_rife_bentoml" in docker_text
    assert "COPY datasets" not in docker_text
    assert "COPY raw_data" not in docker_text
    assert "COPY outputs" not in docker_text
    assert "COPY notebooks" not in docker_text
    assert "COPY model_weights" not in docker_text

    assert "rife-gpu0" in compose_text
    assert "CUDA_VISIBLE_DEVICES: \"0\"" in compose_text
    assert "NVIDIA_VISIBLE_DEVICES: \"0\"" in compose_text
    assert "NVIDIA_DRIVER_CAPABILITIES: compute,utility,video" in compose_text
    assert "RIFE_DEVICE: ${RIFE_DEVICE:-cuda}" in compose_text
    assert "RIFE_CODEC: ${RIFE_CODEC:-h264_nvenc}" in compose_text
    assert "\"3000:3000\"" in compose_text
    assert "${RIFE_SHARED_DIR:-/shared/rife}:/shared/rife" in compose_text
    assert "../../model_weights:/app/model_weights:ro" in compose_text
    assert "../../model_repos:/app/model_repos:ro" in compose_text
    assert "../../configs:/app/configs:ro" in compose_text
    assert "device_ids: [\"0\"]" in compose_text
    assert "capabilities: [gpu]" in compose_text
    assert "http://rife-gpu0:3000" in compose_text


def test_mvp_smoke_client_argument_parsing_and_payload() -> None:
    module = _load_smoke_module()
    args = module.parse_args(
        [
            "--url",
            "http://localhost:3000/interpolate_video",
            "--input-path",
            "/shared/rife/job/input.mp4",
            "--output-path",
            "/shared/rife/job/output.mp4",
            "--interpolation-factor",
            "4",
            "--scale",
            "0.5",
            "--output-playback-mode",
            "slow_motion",
            "--disable-quality-evaluation",
        ]
    )

    payload = module.build_payload(args)

    assert args.url == "http://localhost:3000/interpolate_video"
    assert payload == {"request": {
        "input_path": "/shared/rife/job/input.mp4",
        "output_path": "/shared/rife/job/output.mp4",
        "interpolation_factor": 4,
        "scale": 0.5,
        "output_playback_mode": "slow_motion",
        "quality_evaluation_enabled": False,
    }}


def test_mvp_smoke_client_rejects_invalid_interpolation_factor() -> None:
    module = _load_smoke_module()

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--url",
                "http://localhost:3000/interpolate_video",
                "--input-path",
                "/shared/rife/job/input.mp4",
                "--output-path",
                "/shared/rife/job/output.mp4",
                "--interpolation-factor",
                "5",
            ]
        )


def test_mvp_smoke_client_response_and_output_validation(tmp_path: Path) -> None:
    module = _load_smoke_module()
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"video")

    module.validate_response(
        {"status": "completed", "output_path": str(output_path)},
        expected_output_path=str(output_path),
    )
    module.verify_output_file(str(output_path))

    with pytest.raises(RuntimeError, match="status"):
        module.validate_response({"status": "failed", "output_path": str(output_path)}, expected_output_path=str(output_path))
    with pytest.raises(RuntimeError, match="output_path"):
        module.validate_response({"status": "completed", "output_path": "/wrong.mp4"}, expected_output_path=str(output_path))
    with pytest.raises(RuntimeError, match="does not exist"):
        module.verify_output_file(str(tmp_path / "missing.mp4"))

    empty_output = tmp_path / "empty.mp4"
    empty_output.write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty"):
        module.verify_output_file(str(empty_output))


def test_mvp_bentoml_service_loads_runner_once_and_reuses_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    first_output = tmp_path / "first.mp4"
    second_output = tmp_path / "second.mp4"
    input_path.write_bytes(b"input")
    runner = _FakeRunner(write_output=True)
    monkeypatch.setattr(module, "create_runner", lambda: runner)

    service = module.PracticalRIFEInterpolationService.inner()

    first = service.interpolate_video(
        module.InterpolateVideoRequest(
            input_path=str(input_path),
            output_path=str(first_output),
            interpolation_factor=2,
        )
    )
    second = service.interpolate_video(
        module.InterpolateVideoRequest(
            input_path=str(input_path),
            output_path=str(second_output),
            interpolation_factor=4,
            scale=0.5,
            output_playback_mode="slow_motion",
            quality_evaluation_enabled=False,
            quality_sample_count=7,
        )
    )

    assert runner.load_count == 1
    assert first["status"] == "completed"
    assert first["output_path"] == str(first_output)
    assert second["status"] == "completed"
    assert second["output_path"] == str(second_output)
    assert [call["interpolation_factor"] for call in runner.calls] == [2, 4]
    assert runner.calls[0]["scale"] == 1.0
    assert runner.calls[0]["output_playback_mode"] == "real_time"
    assert runner.calls[0]["enable_quality_evaluation"] is True
    assert runner.calls[0]["quality_sample_count"] == 16
    assert runner.calls[1]["scale"] == 0.5
    assert runner.calls[1]["output_playback_mode"] == "slow_motion"
    assert runner.calls[1]["enable_quality_evaluation"] is False
    assert runner.calls[1]["quality_sample_count"] == 7


def test_mvp_service_contract_surfaces_inference_errors(tmp_path: Path) -> None:
    module = _load_service_module()
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")

    with pytest.raises(InternalServerError, match="Practical-RIFE inference failed"):
        module.run_interpolate_video_request(
            _FailingRunner(),
            {
                "input_path": str(input_path),
                "output_path": str(tmp_path / "output.mp4"),
                "interpolation_factor": 2,
            },
        )


class _FakeRunner:
    def __init__(self, *, write_output: bool, output_bytes: bytes = b"video") -> None:
        self.write_output = write_output
        self.output_bytes = output_bytes
        self.load_count = 0
        self.calls: list[dict[str, object]] = []

    def load(self) -> None:
        self.load_count += 1

    def run(
        self,
        *,
        input_path: Path,
        output_path: Path,
        interpolation_factor: int,
        scale: float,
        output_playback_mode: str,
        enable_quality_evaluation: bool,
        quality_sample_count: int,
    ) -> object:
        self.calls.append(
            {
                "input_path": input_path,
                "output_path": output_path,
                "interpolation_factor": interpolation_factor,
                "scale": scale,
                "output_playback_mode": output_playback_mode,
                "enable_quality_evaluation": enable_quality_evaluation,
                "quality_sample_count": quality_sample_count,
            }
        )
        if self.write_output:
            output_path.write_bytes(self.output_bytes)
        return SimpleNamespace(
            total_elapsed_sec=12.34,
            quality_psnr_mean=31.42,
            quality_ssim_mean=0.948,
        )


class _FailingRunner:
    def load(self) -> None:
        return None

    def run(self, **_kwargs: object) -> object:
        raise RuntimeError("model failed")


def _load_service_module():
    path = Path("services/practical_rife_bentoml/service.py")
    spec = importlib.util.spec_from_file_location("practical_rife_bentoml_mvp_service", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive import guard.
        raise ImportError(f"Could not load service module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_smoke_module():
    path = Path("services/practical_rife_bentoml/smoke_test.py")
    spec = importlib.util.spec_from_file_location("practical_rife_bentoml_smoke_test", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive import guard.
        raise ImportError(f"Could not load smoke module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
