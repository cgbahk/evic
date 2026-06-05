# TODO Check performance difference with/without assert
# TODO May remove dependency on `egg`
# TODO Check and compare model description of https://aclanthology.org/2024.cmcl-1.5/
# TODO Support evaluation mode that evaluate only the first context
from typing import Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from egg.core.interaction import LoggingStrategy


class Participant(NamedTuple):
    """
    A participant in multi-agent communication.

    Each participant has a speaker and listener module.

    Attributes:
        speaker_module: The speaker network that produces messages
        listener_module: The listener network that decodes messages
    """
    speaker_module: nn.Module
    listener_module: nn.Module


def initialize_vision_module(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    assert name == "vit"

    # TODO `module.default_cfg["hf_hub_id"]` has detailed information. How to set it specifically?
    #   Or just report it
    #
    # `vit_base_patch16_384` has `hf_hub_id` value of "timm/vit_base_patch16_384.augreg_in21k_ft_in1k"
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
    # NOTE The first argument of `cross_entropy` is expected to be logits, not probabilities.
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

        # TODO Use random permutation
        # Currently, implementation is using cycles original context, but this doesn't need to be,
        # as long as first entries are 0, 1, ..., n-1. Using cyclic permutation might induce
        # unexpected location bias.

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

        batched_logit = self.fc(batched_cycled_context_feature.reshape(B * C, C * F))
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

        batched_image_embed = self.fc(batched_context_feature.reshape(B * C, self._F))
        batched_context_image_embed = batched_image_embed.view(B, C, E)

        # This is highly coupled with loss calculation
        ret = F.cosine_similarity(
            embedded_cyclic_msg.unsqueeze(2),  # (B, C, 1, E)
            batched_context_image_embed.unsqueeze(1),  # (B, 1, C, E)
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
        # TODO Set policy on this
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
            aux_input={},  # Otherwise error during `dump_interactions`
            receiver_output=batched_similarity,
            message=message.detach(),
            message_length=None,  # TODO
            aux=aux_info,
        )

        return loss.mean(), interaction


class MultiAgentLewisGame(nn.Module):
    """
    Multi-agent communication game where each participant has a sender and receiver.
    
    Participants communicate exhaustively: A's sender can communicate to B's receiver
    for all pairs (A, B), including self-communication (A's sender -> A's receiver).
    
    Args:
        vision_module: Shared frozen vision encoder
        participants: List of Participant namedtuples, one per participant
        train_logging_strategy: Logging strategy for training
        test_logging_strategy: Logging strategy for testing
    """

    def __init__(
        self,
        vision_module: nn.Module,
        participants: list[Participant],
        context_size: int,
        # TODO Set policy on this
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super().__init__()
        self._vision_module = vision_module
        self._participant_count = len(participants)
        self._context_size = context_size
        
        # Unpack participants into separate ModuleLists
        self.speakers = nn.ModuleList([p.speaker_module for p in participants])
        self.listeners = nn.ModuleList([p.listener_module for p in participants])
        
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
        Multi-agent forward pass.

        Args:
            batched_context_image: (B, participant_count, context_size, C, H, W)
                where B is the batch of participant groups
            _labels: None (not used)
            _receiver_input: None (not used)
            _aux_input: None (not used)

        Returns:
            loss: mean loss across all sender-receiver pairs
            interaction: logging information
        """
        assert _labels is None
        assert _receiver_input is None
        assert _aux_input is None

        # batched_context_image shape: (B, P, C, img_C, img_H, img_W)
        assert len(batched_context_image.shape) == 6

        B, P, C, img_C, img_H, img_W = batched_context_image.shape
        assert P == self._participant_count
        assert C == self._context_size

        # Extract features for all participants
        # (B*P*C, img_C, img_H, img_W) -> (B*P*C, feature_dim)
        batched_feature = self._vision_module(
            batched_context_image.view(-1, img_C, img_H, img_W)
        )
        assert len(batched_feature.shape) == 2
        assert batched_feature.shape[0] == B * P * C

        # Reshape to (B, P, C, feature_dim)
        batched_context_feature = batched_feature.view(B, P, C, -1)

        # Each participant produces messages for their context
        # messages[participant] = (B, C, vocab_size)
        messages = [speaker(batched_context_feature[:, p, :, :]) for p, speaker in enumerate(self.speakers)]

        # Compute similarity for all sender-receiver pairs
        # Each listener receives messages from all speakers and computes similarity
        all_similarities = []

        for receiver_idx, listener in enumerate(self.listeners):
            # Listener receives messages from all speakers (including self)
            for sender_idx, message in enumerate(messages):
                # Compute similarity: receiver_idx listens to sender_idx's message
                # message: (B, C, vocab_size)
                # batched_context_feature[:, receiver_idx, :, :]: (B, C, feature_dim)
                similarity = listener(message, batched_context_feature[:, receiver_idx, :, :], None)
                # similarity: (B, C, C)
                all_similarities.append(similarity)

        # Stack all similarities: (P*P, B, C, C)
        all_similarities = torch.stack(all_similarities, dim=0)

        # Reshape to (B, P*P, C, C) for loss calculation
        all_similarities = all_similarities.permute(1, 0, 2, 3)

        # Calculate loss for all pairs
        # Reshape to (B*P*P, C, C) for calculate_loss_from_batch
        B, P2, C_dim, _ = all_similarities.shape
        all_similarities_flat = all_similarities.reshape(B * P2, C_dim, C_dim)

        losses, aux_infos = [], []
        for sim in all_similarities_flat:
            loss, aux = calculate_loss_from_batch(sim.unsqueeze(0))  # Add batch dim
            losses.append(loss.squeeze(0))  # Remove batch dim
            aux_infos.append(aux)

        # losses: (B*P*P, C)
        losses = torch.stack(losses, dim=0)
        mean_loss = losses.mean()

        # Aggregate aux info
        avg_acc = torch.stack([aux['acc'] for aux in aux_infos], dim=0).mean(dim=0)
        aux_info = {'acc': avg_acc}

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        # Aggregate messages for logging
        aggregated_message = torch.stack(messages, dim=1)  # (B, P, C, vocab_size)

        interaction = logging_strategy.filtered_interaction(
            sender_input=batched_context_image,
            receiver_input=_receiver_input,
            labels=_labels,
            aux_input={},
            receiver_output=all_similarities.view(B, P2, C_dim, C_dim),
            message=aggregated_message.detach(),
            message_length=None,
            aux=aux_info,
        )

        return mean_loss, interaction
