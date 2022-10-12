from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import pdb

import torch
import numpy as np
from torch.utils.data import DataLoader

class Evaluator:
    def __init__(self, model, dataset, collate_fn, device=None):
        self.model = model
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.device = 'cpu' if device is None else device

    def evaluate(self, code_type, ppl_type, eval_batch_size):
        mimic_val_dataset = self.dataset
        mimic_val_collator = self.collate_fn
        mimic_val_collator.set_eval_code_type(code_type)
        mimic_val_collator.set_eval_ppl_type(ppl_type)
        dataloader = DataLoader(mimic_val_dataset,
            batch_size=eval_batch_size,
            num_workers=0,
            drop_last=False,
            collate_fn=mimic_val_collator,
            shuffle=False,
            pin_memory=False)

        ppl_list = []
        for batch in dataloader:
            if batch is not None:
                batch = self._prepare_inputs(batch)
                with torch.no_grad():
                    outputs = self.model(**batch)
                batch_ppl = outputs.perplexity
                batch_ppl = batch_ppl.cpu().flatten().tolist()
                ppl_list.extend(batch_ppl)
        ppl_ar = np.array(ppl_list)
        return np.median(ppl_ar)

    def _prepare_inputs(self, data):
        return type(data)(**{k: self._prepare_input(v) for k, v in data.items()})

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.device)
            return data.to(**kwargs)
        return data
