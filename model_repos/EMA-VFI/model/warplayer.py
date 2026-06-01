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


def _make_grid(tenFlow):
    tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device, dtype=tenFlow.dtype).view(
        1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
    tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device, dtype=tenFlow.dtype).view(
        1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
    return torch.cat([tenHorizontal, tenVertical], 1)


def warp(tenInput, tenFlow):
    k = _cache_key_for_grid(tenFlow)
    if k is None:
        tenGrid = _make_grid(tenFlow)
    else:
        if k not in backwarp_tenGrid:
            backwarp_tenGrid[k] = _make_grid(tenFlow)
        tenGrid = backwarp_tenGrid[k]

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (tenGrid + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
