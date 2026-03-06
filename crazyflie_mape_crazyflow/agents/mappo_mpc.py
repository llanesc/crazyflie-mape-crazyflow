"""MAPPO agent with MPC state passthrough.

Subclass of SKRL's MAPPO that stores raw (un-normalized) MPC state in memory
and passes it to the policy via inputs["mpc_state"], bypassing all preprocessors.

The environment must provide info["mpc_state"] = {agent_name: ndarray(N, mpc_state_size)}
in both reset() and step() returns.
"""

from __future__ import annotations

from typing import Any

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.multi_agents.torch.mappo.mappo import compute_gae
from skrl.resources.schedulers.torch import KLAdaptiveLR


class MAPPO_MPC(MAPPO):
    """MAPPO with MPC state support.

    Extends MAPPO to store raw MPC state (e.g., [pos, rpy, vel, drpy]) in the
    replay memory and pass it to the policy as inputs["mpc_state"] without any
    preprocessing/normalization.

    The MPC state is provided by the environment via the info dict:
        info["mpc_state"] = {agent_name: ndarray(N, mpc_state_size)}

    Args:
        mpc_state_size: Dimension of the MPC state vector (default: 12).
        **kwargs: All other arguments forwarded to MAPPO.__init__().
    """

    def __init__(self, *, mpc_state_size: int = 12, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mpc_state_size = mpc_state_size
        self._current_mpc_state: dict[str, torch.Tensor] = {}

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent, adding mpc_state tensor to memory."""
        super().init(trainer_cfg=trainer_cfg)

        # Add mpc_state tensor to each agent's memory
        if self.memories and self._mpc_state_size > 0:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(
                    name="mpc_state", size=self._mpc_state_size, dtype=torch.float32
                )
            self._tensors_names.append("mpc_state")

    def act(
        self,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Compute actions, injecting cached mpc_state into policy inputs."""
        actions = {}
        log_prob = {}
        outputs = {}

        for uid in self.possible_agents:
            inputs = {
                "observations": self._observation_preprocessor[uid](observations[uid]),
                "states": self._state_preprocessor[uid](states[uid]),
            }
            # Inject raw MPC state (no preprocessing)
            if self._mpc_state_size > 0 and uid in self._current_mpc_state:
                inputs["mpc_state"] = self._current_mpc_state[uid]

            # sample random actions
            if timestep < self.cfg.random_timesteps:
                actions[uid], outputs[uid] = self.policies[uid].random_act(inputs, role="policy")

            # sample stochastic actions
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions[uid], outputs[uid] = self.policies[uid].act(inputs, role="policy")
                log_prob[uid] = outputs[uid]["log_prob"]

        self._current_log_prob = log_prob
        return actions, outputs

    def record_transition(
        self,
        *,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        actions: dict[str, torch.Tensor],
        rewards: dict[str, torch.Tensor],
        next_observations: dict[str, torch.Tensor],
        next_states: dict[str, torch.Tensor],
        terminated: dict[str, torch.Tensor],
        truncated: dict[str, torch.Tensor],
        infos: dict[str, Any],
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record transition, storing mpc_state in memory and updating cache from infos."""
        # Call grandparent (MultiAgent) record_transition for reward tracking
        super(MAPPO, self).record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.memories:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            for uid in self.possible_agents:
                # reward shaping
                if self.cfg.rewards_shaper is not None:
                    rewards[uid] = self.cfg.rewards_shaper(rewards[uid], timestep, timesteps)

                # compute values
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor[uid](observations[uid]),
                        "states": self._state_preprocessor[uid](states[uid]),
                    }
                    values, _ = self.values[uid].act(inputs, role="value")
                    values = self._value_preprocessor[uid](values, inverse=True)

                # time-limit (truncation) bootstrapping
                if self.cfg.time_limit_bootstrap[uid]:
                    rewards[uid] += self.cfg.discount_factor[uid] * values * truncated[uid]

                # storage transition in memory (with mpc_state)
                samples = {
                    "observations": observations[uid],
                    "states": states[uid],
                    "actions": actions[uid],
                    "rewards": rewards[uid],
                    "terminated": terminated[uid],
                    "log_prob": self._current_log_prob[uid],
                    "values": values,
                }
                if self._mpc_state_size > 0 and uid in self._current_mpc_state:
                    samples["mpc_state"] = self._current_mpc_state[uid]
                self.memories[uid].add_samples(**samples)

        # Update mpc_state cache from infos (for next act() call)
        # infos["mpc_state"] corresponds to next_observations (post-step state)
        if self._mpc_state_size > 0 and "mpc_state" in infos:
            for uid in self.possible_agents:
                if uid in infos["mpc_state"]:
                    self._current_mpc_state[uid] = torch.as_tensor(
                        infos["mpc_state"][uid], dtype=torch.float32, device=self.device
                    )

    def update(self, *, timestep: int, timesteps: int, uid: str) -> None:
        """Update step with mpc_state support in sampled batches."""
        policy = self.policies[uid]
        value = self.values[uid]
        memory = self.memories[uid]

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor[uid](self._current_next_observations[uid]),
                "states": self._state_preprocessor[uid](self._current_next_states[uid]),
            }
            value.enable_training_mode(False)
            last_values, _ = value.act(inputs, role="value")
            value.enable_training_mode(True)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

        values = memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=memory.get_tensor_by_name("rewards"),
            terminated=memory.get_tensor_by_name("terminated"),
            values=values,
            next_values=last_values,
            discount_factor=self.cfg.discount_factor[uid],
            lambda_coefficient=self.cfg.lambda_[uid],
        )

        memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=True))
        memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
        memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = memory.sample_all(names=self._tensors_names, mini_batches=self.cfg.mini_batches[uid])

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        has_mpc_state = self._mpc_state_size > 0

        # learning epochs
        for epoch in range(self.cfg.learning_epochs[uid]):
            kl_divergences = []

            # mini-batches loop
            for batch in sampled_batches:
                # Unpack with optional mpc_state at the end
                if has_mpc_state:
                    *base_tensors, sampled_mpc_state = batch
                else:
                    base_tensors = batch
                    sampled_mpc_state = None
                (
                    sampled_observations,
                    sampled_states,
                    sampled_actions,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                ) = base_tensors

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor[uid](sampled_observations, train=not epoch),
                        "states": self._state_preprocessor[uid](sampled_states, train=not epoch),
                    }
                    if sampled_mpc_state is not None:
                        inputs["mpc_state"] = sampled_mpc_state

                    _, outputs = policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self.cfg.kl_threshold[uid] and kl_divergence > self.cfg.kl_threshold[uid]:
                        break

                    # compute entropy loss
                    if self.cfg.entropy_loss_scale[uid]:
                        entropy_loss = -self.cfg.entropy_loss_scale[uid] * policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip[uid], 1.0 + self.cfg.ratio_clip[uid]
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _ = value.act(inputs, role="value")

                    if self.cfg.value_clip[uid] > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self.cfg.value_clip[uid],
                            max=self.cfg.value_clip[uid],
                        )
                    value_loss = self.cfg.value_loss_scale[uid] * F.mse_loss(sampled_returns, predicted_values)

                # optimization step
                self.optimizers[uid].zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    policy.reduce_parameters()
                    if policy is not value:
                        value.reduce_parameters()

                if self.cfg.grad_norm_clip[uid] > 0:
                    self.scaler.unscale_(self.optimizers[uid])
                    if policy is value:
                        nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[uid])
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(policy.parameters(), value.parameters()), self.cfg.grad_norm_clip[uid]
                        )

                self.scaler.step(self.optimizers[uid])
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale[uid]:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self.schedulers[uid]:
                if isinstance(self.schedulers[uid], KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.schedulers[uid].step(kl.item())
                else:
                    self.schedulers[uid].step()

        # record data
        self.track_data(
            f"Loss / Policy loss ({uid})",
            cumulative_policy_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
        )
        self.track_data(
            f"Loss / Value loss ({uid})",
            cumulative_value_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
        )
        if self.cfg.entropy_loss_scale[uid]:
            self.track_data(
                f"Loss / Entropy loss ({uid})",
                cumulative_entropy_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
            )

        self.track_data(f"Policy / Standard deviation ({uid})", policy.distribution(role="policy").stddev.mean().item())

        if self.schedulers[uid]:
            self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])
