from typing import (
    List,
    Optional,
    Tuple,
)

import torch

from aqueduct import (
    BaseTask,
    BaseTaskHandler,
    Flow,
    FlowStep,
)
from pipeline import (
    Encoder,
    default_producer,
)


class Task(BaseTask):
    def __init__(
            self,
            image: bytes,
    ):
        super().__init__()
        self.image: Optional[bytes, torch.Tensor] = image
        self.embedding: Optional[torch.Tensor] = None


class PreProcessorHandler(BaseTaskHandler):
    def __init__(self):
        self._model = default_producer.get_pre_proc()

    def handle(self, *tasks: Task):
        for task in tasks:
            task.image = self._model.process(task.image)


class EncoderHandler(BaseTaskHandler):
    def __init__(self, max_batch_size: int = 1):
        self._model: Optional[Encoder] = None
        self.max_batch_size = max_batch_size

    def on_start(self):
        self._model = default_producer.get_encoder()

    def handle(self, *tasks: Task):
        embeddings = self._model.process_list(data=[task.image for task in tasks])
        for embedding, task in zip(embeddings, tasks):
            task.embedding = embedding
            task.image = None


class PostProcessorHandler(BaseTaskHandler):
    def __init__(self):
        self._model = default_producer.get_post_proc()

    def handle(self, *tasks: Task):
        for task in tasks:
            task.embedding = self._model.process(task.embedding)


def get_flow() -> Flow:
    return Flow(
        FlowStep(PreProcessorHandler()),
        FlowStep(EncoderHandler()),
        FlowStep(PostProcessorHandler()),
        metrics_enabled=False,
    )