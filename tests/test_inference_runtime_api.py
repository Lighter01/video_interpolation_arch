import pytest
import torch

from video_interpolation.inference_runtime import (
    FramePairRequest,
    FramePairResult,
    InferenceMode,
    RuntimeBackendKind,
    RuntimeInputs,
    RuntimeOutputs,
    generate_interpolation_timesteps,
    validate_interpolation_factor,
)
from video_interpolation.inference_runtime.backends.base import RuntimeBackend, RuntimeBackendError
from video_interpolation.inference_runtime.backends.torch import TorchRuntimeBackend


def test_fixed_2x_request_defaults_to_middle_timestep_and_runtime_inputs() -> None:
    left = torch.zeros(3, 8, 10)
    right = torch.ones(3, 8, 10)

    request = FramePairRequest(left=left, right=right)

    assert request.mode is InferenceMode.FIXED_2X
    assert request.backend_kind is RuntimeBackendKind.TORCH
    assert request.interpolation_factor == 2
    assert request.timesteps == (0.5,)
    assert request.num_intermediate_frames == 1
    assert request.original_shape == (3, 8, 10)
    assert not request.is_batched

    runtime_inputs = request.to_runtime_inputs()
    assert runtime_inputs.tensors["left"] is left
    assert runtime_inputs.tensors["right"] is right
    assert runtime_inputs.timesteps == (0.5,)
    assert runtime_inputs.metadata["mode"] == "fixed_2x"


def test_fixed_2x_request_rejects_non_two_factor_and_non_middle_timestep() -> None:
    left = torch.zeros(3, 8, 10)
    right = torch.ones(3, 8, 10)

    with pytest.raises(ValueError, match="fixed_2x mode requires interpolation_factor=2"):
        FramePairRequest(left=left, right=right, interpolation_factor=4)

    with pytest.raises(ValueError, match="fixed_2x mode requires exactly one timestep"):
        FramePairRequest(left=left, right=right, timesteps=(0.25,))


@pytest.mark.parametrize(
    ("factor", "expected"),
    [
        (2, (0.5,)),
        (4, (0.25, 0.5, 0.75)),
        (8, (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)),
    ],
)
def test_arbitrary_nx_generates_expected_timesteps(factor: int, expected: tuple[float, ...]) -> None:
    left = torch.zeros(1, 3, 8, 10)
    right = torch.ones(1, 3, 8, 10)

    request = FramePairRequest(
        left=left,
        right=right,
        mode="arbitrary_nx",
        interpolation_factor=factor,
    )

    assert request.mode is InferenceMode.ARBITRARY_NX
    assert request.timesteps == expected
    assert request.num_intermediate_frames == factor - 1
    assert request.is_batched
    assert generate_interpolation_timesteps(factor) == expected


def test_arbitrary_nx_validates_custom_timesteps_against_factor_order_and_range() -> None:
    left = torch.zeros(3, 8, 10)
    right = torch.ones(3, 8, 10)

    request = FramePairRequest(
        left=left,
        right=right,
        mode=InferenceMode.ARBITRARY_NX,
        interpolation_factor=4,
        timesteps=(0.2, 0.5, 0.8),
    )
    assert request.timesteps == (0.2, 0.5, 0.8)

    with pytest.raises(ValueError, match="requires 3 timestep"):
        FramePairRequest(left=left, right=right, mode="arbitrary_nx", interpolation_factor=4, timesteps=(0.5,))

    with pytest.raises(ValueError, match="strictly increasing"):
        FramePairRequest(
            left=left,
            right=right,
            mode="arbitrary_nx",
            interpolation_factor=4,
            timesteps=(0.5, 0.25, 0.75),
        )

    with pytest.raises(ValueError, match="open interval"):
        FramePairRequest(
            left=left,
            right=right,
            mode="arbitrary_nx",
            interpolation_factor=4,
            timesteps=(0.25, 0.5, 1.0),
        )


def test_interpolation_factor_validation_rejects_unsupported_values_and_bool() -> None:
    assert validate_interpolation_factor(2) == 2
    assert validate_interpolation_factor(8) == 8

    for value in (1, 9, True, 2.5):
        with pytest.raises(ValueError, match="interpolation_factor"):
            validate_interpolation_factor(value)


def test_request_rejects_unknown_modes_and_mismatched_frame_shapes() -> None:
    left = torch.zeros(3, 8, 10)
    right = torch.ones(3, 8, 11)

    with pytest.raises(ValueError, match="Unsupported mode"):
        FramePairRequest(left=left, right=torch.ones_like(left), mode="recursive")

    with pytest.raises(ValueError, match="same shape"):
        FramePairRequest(left=left, right=right)

    with pytest.raises(ValueError, match="CHW or NCHW"):
        FramePairRequest(left=torch.zeros(8, 10), right=torch.ones(8, 10))


def test_frame_pair_result_validates_output_count_and_middle_frame() -> None:
    frame = torch.full((3, 8, 10), 0.5)

    result = FramePairResult(
        intermediate_frames=(frame,),
        timesteps=(0.5,),
        mode=InferenceMode.FIXED_2X,
        interpolation_factor=2,
        backend_kind=RuntimeBackendKind.TORCH,
        model_name="unit_model",
        original_shape=frame.shape,
        padded_shape=(1, 3, 8, 16),
    )

    assert result.middle_frame is frame
    assert result.original_shape == (3, 8, 10)
    assert result.padded_shape == (1, 3, 8, 16)

    with pytest.raises(ValueError, match="Expected 3 intermediate frame"):
        FramePairResult(
            intermediate_frames=(frame,),
            timesteps=(0.25, 0.5, 0.75),
            mode="arbitrary_nx",
            interpolation_factor=4,
            backend_kind="torch",
        )

    nx_result = FramePairResult(
        intermediate_frames=(frame, frame, frame),
        timesteps=(0.25, 0.5, 0.75),
        mode="arbitrary_nx",
        interpolation_factor=4,
        backend_kind="torch",
    )
    with pytest.raises(ValueError, match="single-frame"):
        _ = nx_result.middle_frame


def test_runtime_backend_base_enforces_load_before_run_and_close_state() -> None:
    backend = _RecordingBackend()
    inputs = RuntimeInputs(tensors={"input": torch.ones(1)})

    with pytest.raises(RuntimeBackendError, match="not loaded"):
        backend.run(inputs)

    backend.load()
    output = backend.run(inputs)
    assert output.primary_tensor.item() == 2.0
    assert backend.is_loaded

    backend.close()
    assert not backend.is_loaded


def test_torch_runtime_backend_runs_loaded_runner_without_grad_and_normalises_output() -> None:
    module = torch.nn.Linear(1, 1)
    backend = TorchRuntimeBackend(_double_primary_input, module=module)
    inputs = RuntimeInputs(tensors={"input": torch.tensor([2.0], requires_grad=True)})

    with pytest.raises(RuntimeBackendError, match="not loaded"):
        backend.run(inputs)

    backend.load()
    assert not module.training

    output = backend.run(inputs)

    assert output.primary_tensor.item() == 4.0
    assert not output.primary_tensor.requires_grad


class _RecordingBackend(RuntimeBackend):
    kind = RuntimeBackendKind.TORCH

    def __init__(self) -> None:
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        self._is_loaded = True

    def run(self, inputs: RuntimeInputs):
        self.ensure_loaded()
        return RuntimeOutputs(tensors={"output": inputs.primary_tensor + 1})

    def close(self) -> None:
        self._is_loaded = False


def _double_primary_input(inputs: RuntimeInputs) -> torch.Tensor:
    return inputs.primary_tensor * 2
