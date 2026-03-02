import torch
from abc import ABC, abstractmethod
from typing import Union, Optional, List


class AudioTokenizer(ABC):
    @property
    @abstractmethod
    def sampling_rate(self):
        pass

    @property
    @abstractmethod
    def frame_size(self):
        pass

    @abstractmethod
    def encode_from_waveform(
        self,
        waveform: Union[torch.Tensor, List[torch.Tensor]],
        original_sampling_rate: int,
        num_quantizers: Optional[int] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        pass

    @abstractmethod
    def decode_to_waveform(
        self, audio_codes: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        pass

    @abstractmethod
    def set_device(self, device: str):
        pass
