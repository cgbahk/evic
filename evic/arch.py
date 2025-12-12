# TODO Check performance difference with/without assert
# TODO May remove dependency on `egg`
# TODO Check and compare model description of https://aclanthology.org/2024.cmcl-1.5/
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from egg.core.interaction import LoggingStrategy


def initialize_vision_module(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    assert name == "vit"

    module = timm.create_model(
        model_name="vit_base_patch16_384",
        pretrained=pretrained,
    )

    feature_dim = module.head.in_features
    module.head = nn.Identity()

    if pretrained:
        for param in module.parameters():
            param.requires_grad = False
        module = module.eval()

    return module, feature_dim


def calculate_loss_from_batch(batched_sim):  # TODO Find better place
    """
    batched_sim: Batched similarity matrix with (batch, message, context) axes
    Returns:
        loss: (B, C) cross entropy loss per message in each batch
        stats: dict containing accuracy per batch
    """
    B, C, _C = batched_sim.shape
    assert C == _C

    labels = torch.arange(C, device=batched_sim.device).unsqueeze(0).expand(B, C)

    acc = (batched_sim.argmax(dim=-1) == labels).detach().float()
    assert acc.shape == (B, C)  # TODO May need to be flatten

    # Flatten for cross entropy: treat each (batch, message) as an independent "example"
    batched_sim_flat = batched_sim.view(B * C, C)
    labels_flat = labels.reshape(B * C)

    # "input" (first argument) is 2D. "target" (second) is 1D.
    # In this case, `cross_entropy` is calculated for each "example"
    loss_flat = F.cross_entropy(batched_sim_flat, labels_flat, reduction="none")

    loss = loss_flat.view(B, C)
    return loss, {"acc": acc}


class SpeakOnContextInCycle(nn.Module):
    def __init__(
        self,
        context_size: int,
        vision_feature_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self._C = context_size
        self._F = vision_feature_dim
        self._V = vocab_size

        self.fc = nn.Sequential(
            nn.Linear(self._C * self._F, self._V),
            nn.BatchNorm1d(self._V),
        )

    def forward(self, batched_context_feature):
        assert len(batched_context_feature.shape) == 3

        # Batch size, Context size, Feature size
        B, C, F = batched_context_feature.shape
        assert self._C == C and self._F == F

        # For example when C = 3, it would look like:
        #
        #   [[0, 1, 2],
        #    [1, 2, 0],
        #    [2, 0, 1]]
        cyclic_rotation = (
            torch.arange(C).unsqueeze(0) + torch.arange(C).unsqueeze(1)
        ) % C

        # For x[b, :, :   ]  =  [f0,     f1, ... , f(c-1)],
        #
        #     y[b, :, :, :]  = [[f0,     f1, ... , f(c-1)],
        #                       [f1,     f2, ... , f0    ],
        #                       ...
        #                       [f(c-1), f0, ... , f(c-2)]]
        batched_cycled_context_feature = batched_context_feature[:, cyclic_rotation, :]

        assert batched_cycled_context_feature.shape == (B, C, C, F)

        batched_logit = self.fc(batched_cycled_context_feature.view(B * C, C * F))
        assert batched_logit.shape == (B * C, self._V)

        return batched_logit.view(B, C, self._V)


class ListenAndDecideWithContext(nn.Module):
    def __init__(
        self,
        vision_feature_dim: int,
        hidden_dim: int,
        embed_dim: int,
        temperature: float,
    ):
        super().__init__()

        self._F = vision_feature_dim
        self._E = embed_dim

        self._temperature = temperature

        # TODO Simpler architecture may still work?
        self.fc = nn.Sequential(
            nn.Linear(self._F, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self._E, bias=False),
        )

    def forward(self, embedded_cyclic_msg, batched_context_feature, _):
        """
        Args: Need to aligned with `SymbolReceiverWrapper`. Note embedding is done in `SymbolReceiverWrapper`.
        Return: Batched similarity matrix with (batch, message, context) axes
        """
        assert _ is None

        assert len(embedded_cyclic_msg.shape) == 3

        B, C, E = embedded_cyclic_msg.shape
        assert self._E == E
        assert batched_context_feature.shape == (B, C, self._F)

        batched_logit = self.fc(batched_context_feature.view(B * C, self._F))
        batched_context_logit = batched_logit.view(B, C, E)

        # This is highly coupled with loss calculation
        ret = F.cosine_similarity(
            embedded_cyclic_msg.unsqueeze(2),  # (B, C, 1, E)
            batched_context_logit.unsqueeze(1),  # (B, 1, C, E)
            dim=-1,
        )
        ret = ret / self._temperature  # TODO Find better place
        assert ret.shape == (B, C, C)

        return ret


class LewisGameOnImageContext(nn.Module):
    # TODO Consider evaluation
    # TODO Check whether `vision_module` remains freezed indeed

    def __init__(
        self,
        vision_module: nn.Module,
        speaker: nn.Module,
        listener: nn.Module,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super().__init__()
        self._vision_module = vision_module
        self.speaker = speaker
        self.listener = listener
        self.train_logging_strategy = train_logging_strategy
        self.test_logging_strategy = test_logging_strategy

    def forward(
        self,
        batched_context_image,
        _labels,
        _receiver_input,
        _aux_input,
    ):
        """
        I/O structures are defined by `egg.core.Trainer`. But we use a bit different
        convention defined by `egg.Batch`.

        Return: tuple of
        - loss: differentiable loss to be minimized
        - interaction: interaction, or a dictionary (potentially empty) with
            auxiliary metrics that would be aggregated and reported
        """
        assert _labels is None
        assert _receiver_input is None
        assert _aux_input is None

        assert len(batched_context_image.shape) == 5

        B, context_size, C, H, W = batched_context_image.shape
        batched_feature = self._vision_module(batched_context_image.view(-1, C, H, W))

        assert len(batched_feature.shape) == 2
        assert batched_feature.shape[0] == B * context_size

        batched_context_feature = batched_feature.view(B, context_size, -1)

        message = self.speaker(batched_context_feature)
        batched_similarity = self.listener(message, batched_context_feature)
        loss, aux_info = calculate_loss_from_batch(batched_similarity)

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=batched_context_image,
            receiver_input=_receiver_input,
            labels=_labels,
            aux_input=_aux_input,
            receiver_output=batched_similarity,
            message=message.detach(),
            message_length=None,  # TODO
            aux=aux_info,
        )

        return loss.mean(), interaction
