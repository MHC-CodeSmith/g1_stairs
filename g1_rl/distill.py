"""PPO + multi-teacher action distillation for the DWAQ policy ("kickstarting", as in HANDOFF, arXiv 2606.06493).

The env provides, for the observation the student is about to act on, `teacher_actions` (N, A) in the student's raw
action space and `teacher_weights` (N, 2A) >= 0: which teacher is valid for that env and which joints it may
supervise, split into an annealed part [:A] and a fixed part [A:]. The update adds
  coef(it) * L(w[:A]) + fixed_coef * L(w[A:]),   L(w) = sum_j w_j (mu_j - a*_j)^2 / sum_j w_j
to the DWAQ PPO loss. Labels are collected on the student's own state distribution (DAgger) and the task reward stays
in charge. The annealed part lets the student outgrow a teacher it already matches (the warm-start policy); the fixed
part keeps a new skill supervised for the whole run.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import DWAQPPO
from rsl_rl.storage import RolloutStorageDWAQ


class DistillRolloutStorage(RolloutStorageDWAQ):
    class Transition(RolloutStorageDWAQ.Transition):
        def __init__(self):
            super().__init__()
            self.teacher_actions = None
            self.teacher_weights = None

    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_hist_shape,
                 action_shape, device="cpu"):
        super().__init__(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_hist_shape,
                         action_shape, device)
        self.teacher_actions = torch.zeros(num_transitions_per_env, num_envs, *action_shape, device=device)
        self.teacher_weights = torch.zeros(num_transitions_per_env, num_envs, 2 * action_shape[0], device=device)

    def add_transitions(self, transition):
        self.teacher_actions[self.step].copy_(transition.teacher_actions)
        self.teacher_weights[self.step].copy_(transition.teacher_weights)
        super().add_transitions(transition)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """RolloutStorageDWAQ.mini_batch_generator (feed-forward) + (teacher_actions, teacher_weights)."""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        critic = self.privileged_observations if self.privileged_observations is not None else self.observations
        fields = [self.observations, critic, self.prev_critic_obs, self.observation_history, self.actions, self.values,
                  self.advantages, self.returns, self.actions_log_prob, self.mu, self.sigma]
        flat = [f.flatten(0, 1) for f in fields]
        ta, tw = self.teacher_actions.flatten(0, 1), self.teacher_weights.flatten(0, 1)
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                b = indices[i * mini_batch_size:(i + 1) * mini_batch_size]
                obs, crit, prev, hist, act, val, adv, ret, logp, mu, sigma = (f[b] for f in flat)
                # order as the parent yields it: ..., target_values, advantages, returns, ...
                yield (obs, crit, prev, hist, act, val, adv, ret, logp, mu, sigma, (None, None), None, ta[b], tw[b])


class DistillDWAQPPO(DWAQPPO):
    teacher_source = None       # object with .teacher_actions / .teacher_weights for the obs being acted on (the env)
    total_iterations = 1000
    coef_start, coef_end = 1.0, 0.05
    hold_frac, anneal_frac = 0.3, 0.5   # full weight for the first 30% of updates, linear to coef_end over the next 50%
    fixed_coef = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transition = DistillRolloutStorage.Transition()
        self.num_updates = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_hist_shape,
                     action_shape):
        self.storage = DistillRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
                                             obs_hist_shape, action_shape, self.device)

    def act(self, obs, critic_obs, prev_critic_obs, obs_history):
        actions = super().act(obs, critic_obs, prev_critic_obs, obs_history)
        src = self.teacher_source
        self.transition.teacher_actions = src.teacher_actions.to(self.device).clone()
        self.transition.teacher_weights = src.teacher_weights.to(self.device).clone()
        return actions

    def distill_coef(self):
        x = self.num_updates / max(self.total_iterations, 1)
        if x <= self.hold_frac:
            return self.coef_start
        s = min((x - self.hold_frac) / self.anneal_frac, 1.0)
        return self.coef_start + s * (self.coef_end - self.coef_start)

    def update(self, beta: float = 1.0) -> dict[str, float]:
        """DWAQPPO.update (surrogate + value + entropy + beta-VAE) plus the weighted distillation term."""
        coef = self.distill_coef()
        mean_value_loss = mean_surrogate_loss = mean_autoenc_loss = mean_distill = mean_fix = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (obs_batch, critic_obs_batch, prev_critic_obs_batch, obs_hist_batch, actions_batch, target_values_batch,
             advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
             hid_states_batch, masks_batch, teacher_batch, weight_batch) in generator:
            self.policy.act(obs_batch, obs_hist_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch, sigma_batch, entropy_batch = self.policy.action_mean, self.policy.action_std, self.policy.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                                   + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                                   / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent = self.policy.cenet_forward(obs_hist_batch)
            vel_target = critic_obs_batch[:, self.obs_dim:self.obs_dim + 3].detach()
            decode_target = obs_batch[:, :self.obs_dim].detach()
            logvar_latent_clamped = torch.clamp(logvar_latent, min=-10.0, max=10.0)
            kl_divergence = -0.5 * torch.sum(1 + logvar_latent_clamped - mean_latent.pow(2) - logvar_latent_clamped.exp())
            autoenc_loss = (nn.MSELoss()(code_vel, vel_target) + nn.MSELoss()(decode, decode_target)
                            + beta * kl_divergence) / self.num_mini_batches

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((value_batch - returns_batch).pow(2), (value_clipped - returns_batch).pow(2)).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            A = mu_batch.shape[1]
            err2 = (mu_batch - teacher_batch).pow(2)
            w_ann, w_fix = weight_batch[:, :A], weight_batch[:, A:]
            distill_ann = (w_ann * err2).sum() / w_ann.sum().clamp(min=1.0)
            distill_fix = (w_fix * err2).sum() / w_fix.sum().clamp(min=1.0)
            distill_loss = distill_ann + distill_fix

            loss = (surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                    + autoenc_loss + coef * distill_ann + self.fixed_coef * distill_fix)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_autoenc_loss += autoenc_loss.item()
            mean_distill += distill_loss.item()
            mean_fix += distill_fix.item()

        n = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self.num_updates += 1
        self.last_losses = {"distill": mean_distill / n, "distill_coef": coef}
        return {"value_function": mean_value_loss / n, "surrogate": mean_surrogate_loss / n,
                "autoencoder": mean_autoenc_loss / n, "distill": mean_distill / n, "distill_fixed": mean_fix / n,
                "distill_coef": coef}
