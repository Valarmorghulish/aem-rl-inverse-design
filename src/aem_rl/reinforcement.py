"""
Policy-gradient training loop coupling the generator with a reward function.

Implements the simple REINFORCE-style update used in:
    Popova, M., Isayev, O., & Tropsha, A. (2018).
    Deep reinforcement learning for de novo drug design.
    Science Advances, 4(7), eaap7885.

For each policy step the generator emits ``n_batch`` candidate (hydrophilic,
hydrophobic) motif pairs. A pair is *accepted* only if its scalar reward is
strictly positive; otherwise it is resampled. Once ``n_batch`` accepted
samples have been collected, their per-token log-probabilities are weighted
by a discounted reward and aggregated into a single policy-gradient loss.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F


class Reinforcement:
    """REINFORCE-style training driver.

    Parameters
    ----------
    generator : object
        Stack-augmented generator with ``evaluate``, ``forward``,
        ``init_hidden``, ``init_stack``, ``optimizer``, ``device`` etc.
    predictor : object
        Property predictor; passed through to ``get_reward_func``.
    get_reward_func : callable
        ``(hydro_smiles_full, phobic_smiles_full, predictor, **kwargs)
        -> (reward: float, details: dict)``. The reward should be
        non-negative; non-positive rewards are interpreted as filter rejects
        and the corresponding sample is resampled.
    """

    def __init__(
        self,
        generator,
        predictor,
        get_reward_func: Callable[..., Tuple[float, dict]],
    ):
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward_func

    # ---------------------------------------------------------------
    def policy_gradient_step(
        self,
        data,
        n_batch: int,
        gamma: float,
        grad_clipping: float = 1.0,
        max_attempts_per_sample: int = 200,
        **kwargs,
    ) -> Tuple[float, float, List[dict], int]:
        """Run one policy-gradient batch.

        Returns ``(avg_reward, loss, debug_info, total_attempts)`` where
        ``total_attempts`` counts every generated pair (accepted + rejected),
        which is useful for monitoring the acceptance ratio.

        ``max_attempts_per_sample`` guards against pathological reward
        functions that never produce a positive reward; if the limit is hit
        the corresponding slot is filled with a zero-reward placeholder so
        the training loop can move on.
        """
        self.generator.train()
        self.generator.optimizer.zero_grad()

        batch_loss = torch.zeros(1, device=self.generator.device)
        batch_total_reward = 0.0
        batch_debug: List[dict] = []
        batch_attempts = 0

        for _ in range(n_batch):
            reward = 0.0
            details: dict = {}
            hydro_full = ""
            phobic_full = ""
            attempts = 0

            while reward <= 0:
                attempts += 1
                hydro_full = self.generator.evaluate(data, prime_str="^")
                phobic_full = self.generator.evaluate(data, prime_str="~")
                reward, details = self.get_reward(
                    hydro_full, phobic_full, self.predictor, **kwargs
                )
                if attempts >= max_attempts_per_sample:
                    # Give up on this slot; record the last (rejected) sample
                    # but treat it as having zero reward so it does not push
                    # the policy in either direction.
                    reward = 0.0
                    details["stage"] = (
                        f"Skipped: no positive reward after {attempts} attempts"
                    )
                    break

            batch_attempts += attempts
            details["n_attempts"] = attempts
            batch_debug.append(details)
            batch_total_reward += reward

            if reward > 0:
                self._accumulate_loss(hydro_full, reward, gamma, batch_loss, data)
                self._accumulate_loss(phobic_full, reward, gamma, batch_loss, data)

        # Average per (sample, motif). Two motifs (^ and ~) per accepted pair.
        loss = batch_loss / max(2 * n_batch, 1)
        avg_reward = batch_total_reward / max(n_batch, 1)

        loss.backward()
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), grad_clipping
            )
        self.generator.optimizer.step()

        return float(avg_reward), float(loss.item()), batch_debug, batch_attempts

    # ---------------------------------------------------------------
    def _accumulate_loss(
        self,
        full_trajectory: str,
        final_reward: float,
        gamma: float,
        accumulator: torch.Tensor,
        data,
    ) -> None:
        """Accumulate the policy-gradient loss for one trajectory."""
        traj = data.char_tensor(full_trajectory)
        discounted = float(final_reward)

        hidden = self.generator.init_hidden()
        stack = self.generator.init_stack()

        for p in range(len(traj) - 1):
            output, hidden, stack = self.generator.forward(traj[p], hidden, stack)
            log_probs = F.log_softmax(output, dim=1)
            accumulator -= log_probs[0, traj[p + 1]] * discounted
            discounted *= gamma
