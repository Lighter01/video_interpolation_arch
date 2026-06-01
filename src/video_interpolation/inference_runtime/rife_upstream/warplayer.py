import torch

backwarp_tenGrid = {}


def _is_exporting():
    if torch.onnx.is_in_onnx_export():
        return True
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and getattr(compiler, "is_compiling", lambda: False)():
        return True
    return False


def _cache_key_for_grid(tenFlow):
    if _is_exporting():
        return None
    key = (str(tenFlow.device), str(tenFlow.dtype), tenFlow.shape[0], tenFlow.shape[2], tenFlow.shape[3])
    try:
        hash(key)
    except TypeError:
        return None
    return key


def _normalised_axis(length, device, dtype):
    values = torch.arange(length, device=device, dtype=dtype)
    if not _is_exporting() and length == 1:
        return torch.zeros_like(values)
    return values * (2.0 / (length - 1)) - 1.0


def _make_grid(tenFlow):
    tenHorizontal = _normalised_axis(tenFlow.shape[3], tenFlow.device, tenFlow.dtype).view(
        1, 1, 1, tenFlow.shape[3]
    ).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
    tenVertical = _normalised_axis(tenFlow.shape[2], tenFlow.device, tenFlow.dtype).view(
        1, 1, tenFlow.shape[2], 1
    ).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
    return torch.cat([tenHorizontal, tenVertical], 1)


def warp(tenInput, tenFlow):
    key = _cache_key_for_grid(tenFlow)
    if key is None:
        tenGrid = _make_grid(tenFlow)
    else:
        if key not in backwarp_tenGrid:
            backwarp_tenGrid[key] = _make_grid(tenFlow)
        tenGrid = backwarp_tenGrid[key]

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    g = (tenGrid + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
