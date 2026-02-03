"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layer_dims = [state_dim] + list(hidden_dims) + [action_dim * chunk_size]
        layers_list = []
        for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
            layers_list.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])

        self.layers = nn.Sequential(*layers_list[:-1]) #dont want final relu

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action_chunk_pred = self.sample_actions(state)
        B = action_chunk_pred.shape[0]
        loss = 1/B * torch.pow((action_chunk_pred - action_chunk), 2).sum()
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.layers(state).view(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layer_dims = [state_dim + action_dim*chunk_size + 1] + list(hidden_dims) + [action_dim * chunk_size]
        layers_list = []
        for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
            layers_list.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])

        self.layers = nn.Sequential(*layers_list[:-1]) #dont want final relu 

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        device = state.device
        
        action_chunk = action_chunk.view(B, -1)
        action_chunk_noise = torch.randn_like(action_chunk, device=device)
        random_timesteps = torch.rand((B, 1), device=device)

        intermediate_action_chunk = random_timesteps * action_chunk + (1-random_timesteps) * action_chunk_noise
        input = torch.concat([state, intermediate_action_chunk, random_timesteps], dim=1)
        velocity_pred = self.layers(input) #(B, action_chunk_size * action_dim)
        
        velocity_gt = (action_chunk - action_chunk_noise).view(B, self.action_dim*self.chunk_size)
        loss = 1/B * (velocity_pred - velocity_gt).pow(2).sum()
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        device = state.device
        action_chunk = torch.randn((B, self.action_dim * self.chunk_size))

        timesteps = torch.linspace(0, 1, num_steps+1, device=device)[:-1] #dont include the last timestep
        for timestep in timesteps:
            timestep = timestep.unsqueeze(0).repeat(B, 1)
            input = torch.concat([state, action_chunk, timestep], dim=1) 
            velocity = self.layers(input)
            action_chunk = action_chunk + 1/num_steps * velocity
        
        return action_chunk.view(-1, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
