"""Use the saved router to choose which types of information a question needs."""

from pathlib import Path

import torch

from . import common
from .data import ROUTES, ROUTE_LABELS, checkpoint_task, file_hash


def validate_query(query):
    """Check that the question contains some text."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Enter a nonempty question')


class ModernBERTRouter:
    """Use ModernBERT to choose tables, figures, text, or a combination."""
    def __init__(self, checkpoint, device):
        """Load the saved router and keep the settings needed to use it."""
        self.model, self.tokenizer = common.load_checkpoint(checkpoint, torch.device(device))
        self.device = torch.device(device)
        self.provenance = {
            'checkpoint': str(checkpoint),
            'config_sha256': file_hash(Path(checkpoint) / 'config.json'),
            'max_length': self.model.config.router_max_length,
            'model_input_fields': ['query'], 'device': str(self.device),
        }

    @torch.inference_mode()
    def route(self, query):
        """Look at the original question and choose where to search."""
        validate_query(query)
        checkpoint_task(self.model.config)
        inputs = common.encode_queries(self.tokenizer, [query],
                                       self.model.config.router_max_length, self.device)
        with common.mixed_precision(self.device):
            logits = self.model(**inputs).logits.float()[0]
        if logits.shape != (len(ROUTES),) or not torch.isfinite(logits).all():
            raise ValueError('Expected seven finite router logits')
        index = logits.argmax().item()
        return {'route': ROUTES[index].copy(), 'label': ROUTE_LABELS[index],
                'probabilities': dict(zip(ROUTE_LABELS, logits.softmax(-1).cpu().tolist())),
                'provenance': self.provenance}


