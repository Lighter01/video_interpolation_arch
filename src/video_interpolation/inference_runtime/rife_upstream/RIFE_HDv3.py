import torch

from video_interpolation.inference_runtime.rife_upstream.IFNet_HDv3 import IFNet


class Model:
    """Inference-only Practical-RIFE v4.26 model wrapper."""

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self._device = torch.device(device)
        self.flownet = IFNet().to(self._device)
        self.version = 4.26

    def train(self) -> None:
        self.flownet.train()

    def eval(self) -> None:
        self.flownet.eval()

    def device(self) -> None:
        self.flownet.to(self._device)

    def inference(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float | torch.Tensor = 0.5,
        scale: float = 1.0,
    ) -> torch.Tensor:
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        _, _, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]
