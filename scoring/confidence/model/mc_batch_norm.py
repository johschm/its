from torch_uncertainty.post_processing.mc_batch_norm import MCBatchNorm
from confidence.base_confidence import ConfidenceModule
import torch
from typing import Optional, Callable, Any

from confidence.model.base_model import ModelBasedConfidence


class MonteCarloBatchNormConfidence(ModelBasedConfidence):
    def __init__(
        self,
        model: torch.nn.Module,
        confidence: ConfidenceModule,
        num_estimators: int = 16,
        convert: bool = True,
        mc_batch_size: int = 32,
        device: Optional[torch.device] = None,
        average: bool = True,
        index: Optional[int] = None,
        softmax: bool = False,   # renamed from softmax_before_average -> softmax
    ):
        super().__init__(model,confidence, index=index)
        self.estimators = num_estimators
        self.average = average
        self.softmax = softmax
        # MCBatchNorm returns raw logits for each estimator
        self.mc = MCBatchNorm(model, num_estimators, convert, mc_batch_size, device).eval()
        self.confidence = confidence
        self.fitted = False

    def fit(self, dataloader):
        self.mc.fit(dataloader)
        self.fitted = True
        return self

    def forward(self, x, y=None):
        if not self.fitted:
            raise RuntimeError("The MCBatchNorm model has not been fitted. Call `fit` with a dataloader first.")
        B = x.size(0)
        # ensure counters start fresh
        self.mc.reset_counters()
        # collect E pytree outputs (each leaf shape (B, C) per estimator)
        outputs = [self.mc._est_forward(x) for _ in range(self.estimators)]
        # stack each leaf along dim=1 -> shape (B, E, ...)
        batched = torch.utils._pytree.tree_map(lambda *ts: torch.stack(ts, dim=1), *outputs)

        # compute mean logits (always available) to be returned as the "logits" output
        mean_logits = torch.utils._pytree.tree_map(lambda t: t.mean(dim=1), batched)

        # Build a separate object that will be passed to the confidence module:
        # - If average==True and softmax==True -> pass mean_probs (softmax applied to mean logits OR softmax before average? we choose softmax-before-mean of samples to respect "softmax enabled" semantics for MC)
        # - If average==True and softmax==False -> pass mean_logits (averaged logits)
        # - If average==False and softmax==True -> pass per-sample probs (softmax applied to each sample)
        # - If average==False and softmax==False -> pass per-sample logits (batched)
        if self.average:
            if self.softmax:
                # softmax each sample then mean the probabilities
                conf_input = torch.utils._pytree.tree_map(lambda t: torch.nn.functional.softmax(t, dim=-1).mean(dim=1), batched)
            else:
                # mean logits (no softmax) -> energy-style usage or other logits-based criterion
                conf_input = mean_logits
            output = mean_logits  # always return logits for consistency
        else:
            # keep full MC dimension
            if self.softmax:
                conf_input = torch.utils._pytree.tree_map(lambda t: torch.nn.functional.softmax(t, dim=-1), batched)
            else:
                conf_input = batched
            output = batched  # stacked logits

        # compute confidence using the prepared conf_input (probs or logits depending on softmax flag)
        conf = self.confidence(conf_input, y)

        if self.index is None:
            return conf, output
        else:
            # index into each leaf (keep semantics consistent with previous implementation)
            return conf, output[self.index]