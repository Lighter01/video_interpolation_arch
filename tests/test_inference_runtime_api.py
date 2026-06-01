import pytest
import torch

from video_interpolation.inference_runtime import (
    FramePairRequest,
    FramePairResult,
    InferenceMode,
    ModelBatchRequest,
    ModelBatchResult,
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


def test_model_batch_request_accepts_bchw_inputs_and_rejects_invalid_shapes() -> None:
    left = torch.zeros(2, 3, 8, 10)
    right = torch.ones(2, 3, 8, 10)

    request = ModelBatchRequest(left=left, right=right)

    assert request.pair_count == 2
    assert request.batch_size == 2
    assert request.timesteps_per_pair == 1
    assert request.num_intermediate_frames == 1
    assert request.flattened_size == 2
    assert request.original_shape == (2, 3, 8, 10)

    with pytest.raises(ValueError, match="BCHW"):
        ModelBatchRequest(left=torch.zeros(3, 8, 10), right=torch.ones(3, 8, 10))

    with pytest.raises(ValueError, match="batch sizes"):
        ModelBatchRequest(left=torch.zeros(2, 3, 8, 10), right=torch.ones(3, 3, 8, 10))

    with pytest.raises(ValueError, match="same shape"):
        ModelBatchRequest(left=torch.zeros(2, 3, 8, 10), right=torch.ones(2, 3, 8, 11))


@pytest.mark.parametrize(
    ("factor", "expected"),
    [
        (2, (0.5,)),
        (4, (0.25, 0.5, 0.75)),
        (8, (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)),
    ],
)
def test_model_batch_request_generates_nx_timesteps(factor: int, expected: tuple[float, ...]) -> None:
    request = ModelBatchRequest(
        left=torch.zeros(2, 3, 8, 10),
        right=torch.ones(2, 3, 8, 10),
        mode=InferenceMode.ARBITRARY_NX,
        interpolation_factor=factor,
    )

    assert request.timesteps == expected
    assert request.timesteps_per_pair == factor - 1
    assert request.flattened_size == 2 * (factor - 1)


def test_model_batch_request_reuses_interpolation_factor_validation() -> None:
    left = torch.zeros(2, 3, 8, 10)
    right = torch.ones(2, 3, 8, 10)

    with pytest.raises(ValueError, match="interpolation_factor"):
        ModelBatchRequest(left=left, right=right, mode="arbitrary_nx", interpolation_factor=9)

    with pytest.raises(ValueError, match="fixed_2x mode requires interpolation_factor=2"):
        ModelBatchRequest(left=left, right=right, interpolation_factor=4)


def test_model_batch_flattening_is_pair_major_timestep_minor() -> None:
    left = torch.tensor([[[[10.0]]], [[[20.0]]]])
    right = torch.tensor([[[[100.0]]], [[[200.0]]]])
    request = ModelBatchRequest(left=left, right=right, mode="arbitrary_nx", interpolation_factor=4)

    flattened = request.flatten_pair_timesteps()

    assert flattened.left.shape == (6, 1, 1, 1)
    assert flattened.right.shape == (6, 1, 1, 1)
    assert flattened.left[:, 0, 0, 0].tolist() == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    assert flattened.right[:, 0, 0, 0].tolist() == [100.0, 100.0, 100.0, 200.0, 200.0, 200.0]
    assert flattened.timesteps == (0.25, 0.5, 0.75, 0.25, 0.5, 0.75)
    assert [(index.pair_index, index.timestep_index) for index in flattened.indices] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]

    runtime_inputs = flattened.to_runtime_inputs()
    assert runtime_inputs.metadata["flattening_order"] == "pair_major_timestep_minor"
    assert runtime_inputs.metadata["flat_pair_indices"] == (0, 0, 0, 1, 1, 1)
    assert runtime_inputs.metadata["flat_timestep_indices"] == (0, 1, 2, 0, 1, 2)
    assert runtime_inputs.metadata["source_batch_shape"] == (2, 1, 1, 1)


def test_model_batch_result_reconstructs_outputs_by_pair_then_timestep() -> None:
    request = ModelBatchRequest(
        left=torch.zeros(2, 3, 8, 10),
        right=torch.ones(2, 3, 8, 10),
        mode="arbitrary_nx",
        interpolation_factor=4,
    )
    flattened_outputs = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1)

    result = ModelBatchResult.from_flattened_outputs(flattened_outputs, request, model_name="unit_model")

    assert result.model_name == "unit_model"
    assert result.pair_count == 2
    assert result.timesteps_per_pair == 3
    assert result.outputs[0][0].item() == 0.0
    assert result.outputs[0][1].item() == 1.0
    assert result.outputs[0][2].item() == 2.0
    assert result.outputs[1][0].item() == 3.0
    assert result.outputs[1][1].item() == 4.0
    assert result.outputs[1][2].item() == 5.0
    assert [frame.item() for frame in result.flattened_outputs] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert result.metadata["flattening_order"] == "pair_major_timestep_minor"


def test_model_batch_result_validates_flattened_output_count_and_nested_shape() -> None:
    request = ModelBatchRequest(left=torch.zeros(2, 3, 8, 10), right=torch.ones(2, 3, 8, 10))

    with pytest.raises(ValueError, match="batch size"):
        ModelBatchResult.from_flattened_outputs(torch.zeros(3, 3, 8, 10), request)

    with pytest.raises(ValueError, match="outputs\\[0\\]\\[0\\] must be a CHW tensor"):
        ModelBatchResult(
            outputs=((torch.zeros(1, 3, 8, 10),),),
            timesteps=(0.5,),
            mode="fixed_2x",
            interpolation_factor=2,
            backend_kind="torch",
        )


def test_model_batch_fixed_2x_keeps_one_output_per_pair() -> None:
    request = ModelBatchRequest(left=torch.zeros(3, 3, 8, 10), right=torch.ones(3, 3, 8, 10))

    flattened = request.flatten_pair_timesteps()
    result = ModelBatchResult.from_flattened_outputs(torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1), request)

    assert request.mode is InferenceMode.FIXED_2X
    assert flattened.left.shape == request.left.shape
    assert flattened.timesteps == (0.5, 0.5, 0.5)
    assert [(index.pair_index, index.timestep_index) for index in flattened.indices] == [(0, 0), (1, 0), (2, 0)]
    assert [frame.item() for frame in result.middle_frames] == [0.0, 1.0, 2.0]
    assert result.outputs[2][0].item() == 2.0


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
