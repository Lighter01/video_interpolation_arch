from pathlib import Path

import numpy as np
from PIL import Image
import torch


def tensor_to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().clamp(0.0, 1.0).numpy()
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("Expected a single-image batch")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected a CHW image tensor, got shape {tuple(tensor.shape)}")
    if array.shape[0] not in (1, 3):
        raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(tensor.shape)}")
    array = np.moveaxis(array, 0, -1)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return (array * 255.0).round().astype(np.uint8)


def uint8_hwc_to_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 2:
        array = np.expand_dims(array, axis=2)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] != 3:
        raise ValueError(f"Expected RGB image array, got shape {array.shape}")
    return torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()


def save_triplet_prediction_set(
    output_dir: Path,
    prediction_name: str,
    sample_id: str,
    left: torch.Tensor,
    middle: torch.Tensor,
    prediction: torch.Tensor,
    right: torch.Tensor,
) -> str:
    safe_sample_id = sample_id.replace("/", "_").replace("\\", "_")
    relative_dir = Path("sample_predictions") / prediction_name / safe_sample_id
    output_dir_path = output_dir / relative_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_hwc(left)).save(output_dir_path / "im1.png")
    Image.fromarray(tensor_to_uint8_hwc(middle)).save(output_dir_path / "im2_gt.png")
    Image.fromarray(tensor_to_uint8_hwc(prediction)).save(output_dir_path / "im2_generated.png")
    Image.fromarray(tensor_to_uint8_hwc(right)).save(output_dir_path / "im3.png")
    return relative_dir.as_posix()
