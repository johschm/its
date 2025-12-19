from typing import Optional
import torch
from confidence.base_confidence import ModelBasedConfidence, ConfidenceModule
from model.pointnet_plus import SAModule
import torch.utils._pytree as pytree

class MonteCarloSAModuleConfidence(ModelBasedConfidence):
    def __init__(
        self,
        model: torch.nn.Module,
        confidence: ConfidenceModule,
        samples: int = 4,
        average: bool = True,
        index: Optional[int] = None,
    ):
        super().__init__(model, confidence, index=index)
        self.samples = samples
        self.average = average
        self.enable()
        #raise warning to disable random_start in SAModule later
        warning_msg = (
            "Monte Carlo sampling is enabled. "
            "Make sure to disable random_start in later to avoid affecting other confidence modules."
        )
        print(warning_msg)


    def enable(self):
        """
        Turn on stochastic sampling by setting all SAModule.random_start to True
        and putting the model into eval mode.
        """
        self.model.eval()
        if self.samples > 1:
            for m in self.model.modules():
                if isinstance(m, SAModule):
                    m.random_start = True
        return self

    def disable(self):
        """
        Turn off stochastic sampling by setting all SAModule.random_start to False.
        """
        for m in self.model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
        return self

    from typing import Optional
    import torch
    from confidence.base_confidence import ModelBasedConfidence, ConfidenceModule
    from model.pointnet_plus import SAModule

    class MonteCarloSAModuleConfidence(ModelBasedConfidence):
        def __init__(
                self,
                model: torch.nn.Module,
                confidence: ConfidenceModule,
                samples: int = 4,
                average: bool = True,
                index: Optional[int] = None,
        ):
            super().__init__(model, confidence, index=index)
            self.samples = samples
            self.average = average
            self.disable()

        def enable(self):
            """
            Turn on stochastic sampling by setting all SAModule.random_start to True
            and putting the model into eval mode.
            """
            self.model.eval()
            if self.samples > 1:
                for m in self.model.modules():
                    if isinstance(m, SAModule):
                        m.random_start = True
            return self

        def disable(self):
            """
            Turn off stochastic sampling by setting all SAModule.random_start to False.
            """
            for m in self.model.modules():
                if isinstance(m, SAModule):
                    m.random_start = False
            return self

        @staticmethod
        def _aggregate_outputs(outputs, dim=0):
            first = outputs[0]
            if isinstance(first, torch.Tensor):
                stacked = torch.stack(outputs, dim=dim)
                return stacked.mean(dim=dim)
            elif isinstance(first, tuple):
                return tuple(
                    MonteCarloSAModuleConfidence._aggregate_outputs([o[i] for o in outputs], dim)
                    for i in range(len(first))
                )
            elif isinstance(first, list):
                return [
                    MonteCarloSAModuleConfidence._aggregate_outputs([o[i] for o in outputs], dim)
                    for i in range(len(first))
                ]
            elif isinstance(first, dict):
                return {
                    k: MonteCarloSAModuleConfidence._aggregate_outputs([o[k] for o in outputs], dim)
                    for k in first
                }
            else:
                raise TypeError(f"Unsupported output type: {type(first)}")

        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
            samples = [self.model(x) for _ in range(self.samples)]

            if self.average:
                # your existing aggregation logic...
                output = self._aggregate_outputs(samples, dim=0)
            else:
                # stack in the sample dimension (dim=1) directly
                output = pytree.tree_map(lambda *ts: torch.stack(ts, dim=1), *samples)

            conf = self.confidence(output, y)
            if self.index is not None:
                output = output[self.index]
            return conf, output