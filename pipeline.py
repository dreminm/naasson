import io
import json
import math
import pathlib
import logging
import PIL
import torch

from contextlib import suppress
from dataclasses import dataclass
from typing import List, Tuple, Union
from PIL import Image
from torchvision.models import resnet50
from torchvision import transforms

DEVICE = 'mps'  # can be 'cuda:0'
logger = logging.getLogger()


class PreProcessor:
    def __init__(self, input_size: Union[int, Tuple[int, ...]]):
        self.im_size = input_size if isinstance(input_size, int) else input_size[1]
        self.transforms = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def process(self, im_bytes: bytes) -> torch.Tensor:
        im_pil = Image.open(io.BytesIO(im_bytes)).resize((self.im_size, self.im_size), PIL.Image.LANCZOS)
        return self.transforms(im_pil)


class Encoder:
    def __init__(self, device: str, use_amp: bool):
        self.device = device
        self.amp_autocast = torch.cuda.amp.autocast if use_amp else suppress
        logger.error('model initializing is started')
        self.model = self._get_model()
        logger.error('model initializing is finished')

    def process_list(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        batch = torch.stack(data)
        return [elem for elem in self._process_batch(batch)]

    def _get_model(self):
        model = resnet50(pretrained=True)
        model.to(device=self.device)
        model.eval()
        return model

    def _process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(device=self.device)
        logging.warning(batch.shape)
        with self.amp_autocast(), torch.no_grad():
            result = self.model(batch)
            result = result.detach().cpu()
        logging.warning(result.shape)
        return result


class PostProcessor:
    def __init__(self, use_pca: bool):
        self.use_pca = use_pca

    def process(self, embedding: torch.Tensor) -> List[Tuple[str, float]]:
        if self.use_pca:
            raise NotImplemented
        return embedding.numpy()


class Pipeline:
    def __init__(self, preprocessor: PreProcessor, encoder: Encoder,
                 postprocessor: PostProcessor):
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.postprocessor = postprocessor

    def process(self, im: bytes) -> List[Tuple[str, float]]:
        im_torch = self.preprocessor.process(im)
        embedding = self.encoder.process_list(data=[im_torch])
        processed_embeddings = self.postprocessor.process(embedding)
        return processed_embeddings


class ModelsProducer:

    def get_encoder(self) -> Encoder:
        return Encoder(
            device=DEVICE,
            use_amp=False
        )

    def get_pre_proc(self) -> PreProcessor:
        return PreProcessor((3, 224, 224))

    def get_post_proc(self) -> PostProcessor:
        return PostProcessor(use_pca=False)


default_producer = ModelsProducer()