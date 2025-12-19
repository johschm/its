import torch

from confidence.base_confidence import ConfidenceModule
import torch.nn.functional as F

class NNGuideSplitConfidence(ConfidenceModule):


        def __init__(self, base_confidence: ConfidenceModule, k: int = 10):
            super().__init__()
            self.base_confidence = base_confidence
            self.k = k
            self.feature_index = 0
            self.conf_index = 1


        def fit(self, data, y=None):
            # 2) compute base scores for the bank
            bank_scores = self.base_confidence.forward(data[self.conf_index], y).view(-1)
            # 3) normalize and store features + their scores
            bank_feats = F.normalize(data[self.feature_index], p=2, dim=1)
            self.register_buffer("bank_features", bank_feats)
            self.register_buffer("bank_confidences", bank_scores)
            return self


        def forward(self, x, y=None):
            features = x[self.feature_index]
            oth = x[self.conf_index]
            # 1) compute base scores for test samples
            s = self.base_confidence.forward(oth, y).view(-1)
            # 2) normalize test features
            z_norm = F.normalize(features, p=2, dim=1)

            # Calculate cosine similarities first
            sims = z_norm @ self.bank_features.t()

            # Scale the similarities by bank confidences
            scaled_sims = sims * self.bank_confidences.unsqueeze(0)

            # Take the top-k of the scaled similarities and average them for guidance
            guidance = scaled_sims.topk(self.k, dim=1).values.mean(dim=1)

            # 3) return guided score
            return s * guidance

class SplitConfidence(ConfidenceModule):
    """
    This class computes confidence scores by combining the outputs of two confidence modules.
    This considers the output of the model as a tuple of (intermediate_output, logits) and forwards
    them to the respective confidence modules. The final confidence score is computed by either multiplying
    or adding the outputs of the two confidence modules, depending on the mult parameter.

    """
    def __init__(self, confidence_inter, confidence_final, mult=True, a=1, b=1, scale_inter=None,scale_final=None,avg_inter=None):
        """
        Initializes the SplitConfidence class. T
        Args:
            confidence_inter: The confidence module for the intermediate output.
            confidence_final: The confidence module for the final output.
            mult: If True, the outputs of the two confidence modules are multiplied.
            a: Multiplier for the intermediate confidence module. Ignored if mult is True.
            b: Multiplier for the final confidence module. Ignored if mult is True.
        """
        super(SplitConfidence, self).__init__()
        self.confidence_inter = confidence_inter
        self.confidence_final = confidence_final
        self.mult = mult
        self.a = a
        self.b = b
        if scale_inter is not None:
            if scale_inter == "exp":
                self.scale_inter = torch.exp
            elif scale_inter == "softplus":
                self.scale_inter = torch.nn.functional.softplus
            elif callable(scale_inter):
                # If scale_inter is a callable function, use it directly
                self.scale_inter = scale_inter
            else:
                raise ValueError(f"Unknown scale_inter: {scale_inter}. Use 'exp' or 'softplus'.")
        else:
            self.scale_inter = lambda x,dim=None: x  # No scaling by default

        if scale_final is not None:
            if scale_final == "exp":
                self.scale_final = torch.exp
            elif scale_final == "softplus":
                self.scale_final = torch.nn.functional.softplus
            elif callable(scale_final):
                # If scale_final is a callable function, use it directly
                self.scale_final = scale_final
            else:
                raise ValueError(f"Unknown scale_final: {scale_final}. Use 'exp' or 'softplus'.")
        else:
            self.scale_final = lambda x,dim=None: x

        self.avg_inter = avg_inter

    def forward(self, x, y=None):
        """
        Computes confidence scores by combining the outputs of two confidence modules.

        Args:
            x: Input tensor for the model. This should be a tuple of (intermediate_output, logits).
                - intermediate_output: Shape: (*batch_dims, ...)
                - logits: Shape: (*batch_dims, num_classes)
            y: Optional labels for modules that use them.

        Returns:
            confidence: Confidence scores. Shape: (*batch_dims)
        """
        intermediate_output, logits = x



        # Calculate trust confidence using intermediate representation and predicted classes
        confidence_inter = self.confidence_inter(intermediate_output, None)

        # Calculate final confidence using logits
        confidence_final = self.confidence_final(logits, None)

        # Combine confidences
        if self.mult:
            confidence = confidence_inter * confidence_final
            return confidence
        else:
            confidence = self.a * confidence_inter + self.b * confidence_final
            return confidence


class PredictedSplitConfidence(ConfidenceModule):
    """
    This class computes confidence scores by combining the outputs of two confidence modules,
    where the first module calculates the confidence on both the predicted class(from logits)
     and the intermediate output,

    The output of the model is considered as a tuple of (intermediate_output, logits).
    The predicted class is determined from the logits and passed along with the
    intermediate_output to the trust confidence module.
    """
    def __init__(self, confidence_trust, confidence_final, mult=True, a=1, b=1,predict_inter=True,predict_final=True):
        """
        Initializes the TrustSplitConfidence class.

        Args:
            confidence_trust: The confidence module for trust scores that takes
                              both intermediate output and predicted class.
            confidence_final: The confidence module for the final output.
            mult: If True, the outputs of the two confidence modules are multiplied.
            a: Multiplier for the trust confidence module. Ignored if mult is True.
            b: Multiplier for the final confidence module. Ignored if mult is True.
        """
        super(PredictedSplitConfidence, self).__init__()
        self.confidence_trust = confidence_trust
        self.confidence_final = confidence_final
        self.mult = mult
        self.a = a
        self.b = b
        self.predict_inter = predict_inter
        self.predict_final = predict_final

    def forward(self, x, y=None):
        """
        Computes confidence scores by combining the outputs of two confidence modules.

        Args:
            x: Input tensor for the model. This should be a tuple of (intermediate_output, logits).
                - intermediate_output: Shape: (*batch_dims, ...)
                - logits: Shape: (*batch_dims, num_classes)
            y: Optional labels for modules that use them.

        Returns:
            confidence: Confidence scores. Shape: (*batch_dims)
        """
        intermediate_output, logits = x

        # Get predicted classes from logits
        predicted_classes = torch.argmax(logits, dim=-1)


        # Combine confidences
        if self.mult:
            # Calculate trust confidence using intermediate representation and predicted classes
            confidence_trust = self.confidence_trust(intermediate_output,
                                                     predicted_classes if self.predict_inter else y)

            # Calculate final confidence using logits
            confidence_final = self.confidence_final(logits, predicted_classes if self.predict_final else y)

            confidence = confidence_trust * confidence_final
            return confidence
        else:
            if self.a==0:
                return self.confidence_final(logits, predicted_classes if self.predict_final else y)*self.b
            if self.b==0:
                return self.confidence_trust(intermediate_output,
                                                     predicted_classes if self.predict_inter else y)*self.a

            # Calculate trust confidence using intermediate representation and predicted classes
            confidence_trust = self.confidence_trust(intermediate_output,
                                                     predicted_classes if self.predict_inter else y)

            # Calculate final confidence using logits
            confidence_final = self.confidence_final(logits, predicted_classes if self.predict_final else y)

            confidence = self.a * confidence_trust + self.b * confidence_final
            return confidence


class TrueLabelSplitConfidence(ConfidenceModule):
    """
    This class computes confidence scores by combining the outputs of two confidence modules,
    where the first module calculates the confidence on both the true class(from labels)
     and the intermediate output,

    The output of the model is considered as a tuple of (intermediate_output, logits).
    The true class is determined from the labels and passed along with the
    intermediate_output to the trust confidence module.

    This classes passes the true y to both confidence modules.
    """
    def __init__(self, confidence_trust, confidence_final, mult=True, a=1, b=1):
        """
        Initializes the TrustSplitConfidence class.

        Args:
            confidence_trust: The confidence module for trust scores that takes
                              both intermediate output and true class.
            confidence_final: The confidence module for the final output.
            mult: If True, the outputs of the two confidence modules are multiplied.
            a: Multiplier for the trust confidence module. Ignored if mult is True.
            b: Multiplier for the final confidence module. Ignored if mult is True.
        """
        super(TrueLabelSplitConfidence, self).__init__()
        self.confidence_trust = confidence_trust
        self.confidence_final = confidence_final
        self.mult = mult
        self.a = a
        self.b = b

    def forward(self, x, y=None):
        """
        Computes confidence scores by combining the outputs of two confidence modules.

        Args:
            x: Input tensor for the model. This should be a tuple of (intermediate_output, logits).
                - intermediate_output: Shape: (*batch_dims, ...)
                - logits: Shape: (*batch_dims, num_classes)
            y: Optional labels for modules that use them.

        Returns:
            confidence: Confidence scores. Shape: (*batch_dims)
        """
        intermediate_output, logits = x

        # Combine confidences
        if self.mult:
            # Calculate trust confidence using intermediate representation and true classes
            confidence_trust = self.confidence_trust(intermediate_output, y)

            # Calculate final confidence using logits
            confidence_final = self.confidence_final(logits, y)

            confidence = confidence_trust * confidence_final
            return confidence
        else:
            if self.a==0:
                return self.confidence_final(logits, y)*self.b
            if self.b==0:
                return self.confidence_trust(intermediate_output, y)*self.a

            # Calculate trust confidence using intermediate representation and true classes
            confidence_trust = self.confidence_trust(intermediate_output, y)

            # Calculate final confidence using logits
            confidence_final = self.confidence_final(logits, y)

            confidence = self.a * confidence_trust + self.b * confidence_final
            return confidence
