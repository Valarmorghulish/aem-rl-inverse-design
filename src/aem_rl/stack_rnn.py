"""
Stack-augmented recurrent generator with a differentiable external stack.

Implements the architecture used in:
    Popova, M., Isayev, O., & Tropsha, A. (2018).
    Deep reinforcement learning for de novo drug design.
    Science Advances, 4(7), eaap7885.

The differentiable stack design is from:
    Joulin, A., & Mikolov, T. (2015). Inferring algorithmic patterns with
    stack-augmented recurrent nets. NeurIPS.
"""

from __future__ import annotations

import os
import time
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from .smiles_enumerator import SmilesEnumerator
from .utils import time_since


class StackAugmentedRNN(nn.Module):
    """Stack-augmented GRU/LSTM character-level generator."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        layer_type: str = "GRU",
        n_layers: int = 1,
        is_bidirectional: bool = False,
        has_stack: bool = False,
        stack_width: Optional[int] = None,
        stack_depth: Optional[int] = None,
        use_cuda: Optional[bool] = None,
        optimizer_instance: type = torch.optim.Adadelta,
        lr: float = 0.01,
    ):
        super().__init__()
        if layer_type not in ("GRU", "LSTM"):
            raise ValueError("layer_type must be 'GRU' or 'LSTM'")
        self.layer_type = layer_type
        self.is_bidirectional = is_bidirectional
        self.num_dir = 2 if is_bidirectional else 1
        self.has_cell = layer_type == "LSTM"
        self.has_stack = has_stack

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        if has_stack:
            self.stack_width = int(stack_width)
            self.stack_depth = int(stack_depth)

        self.use_cuda = (
            torch.cuda.is_available() if use_cuda is None else use_cuda
        )
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if has_stack:
            self.stack_controls_layer = nn.Linear(
                hidden_size * self.num_dir, 3
            )
            self.stack_input_layer = nn.Linear(
                hidden_size * self.num_dir, self.stack_width
            )

        self.encoder = nn.Embedding(input_size, hidden_size)
        rnn_input_size = hidden_size + (self.stack_width if has_stack else 0)

        rnn_cls = nn.LSTM if layer_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            rnn_input_size,
            hidden_size,
            n_layers,
            bidirectional=is_bidirectional,
        )

        self.decoder = nn.Linear(hidden_size * self.num_dir, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_instance = optimizer_instance
        self.optimizer = optimizer_instance(
            self.parameters(), lr=lr, weight_decay=1e-5
        )
        self.lr = lr

    # --------------------------- I/O --------------------------------------
    def load_model(self, path: str) -> None:
        weights = torch.load(path, map_location=self.device)
        self.load_state_dict(weights)
        self.to(self.device)

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def change_lr(self, new_lr: float) -> None:
        self.optimizer = self.optimizer_instance(
            self.parameters(), lr=new_lr, weight_decay=1e-5
        )
        self.lr = new_lr

    # --------------------------- forward ----------------------------------
    def forward(self, inp, hidden, stack):
        inp = self.encoder(inp.view(1, -1))
        if self.has_stack:
            hidden_for_stack = hidden[0] if self.has_cell else hidden
            if self.is_bidirectional:
                hidden_2_stack = torch.cat(
                    (hidden_for_stack[0], hidden_for_stack[1]), dim=1
                )
            else:
                hidden_2_stack = hidden_for_stack.squeeze(0)
            stack_controls = F.softmax(
                self.stack_controls_layer(hidden_2_stack), dim=1
            )
            stack_input = torch.tanh(
                self.stack_input_layer(hidden_2_stack.unsqueeze(0))
            )
            stack = self._stack_augmentation(
                stack_input.permute(1, 0, 2), stack, stack_controls
            )
            stack_top = stack[:, 0, :].unsqueeze(0)
            inp = torch.cat((inp, stack_top), dim=2)

        output, next_hidden = self.rnn(inp.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, next_hidden, stack

    def _stack_augmentation(self, input_val, prev_stack, controls):
        batch_size = prev_stack.size(0)
        controls = controls.view(-1, 3, 1, 1)
        zeros_bottom = torch.zeros(
            batch_size, 1, self.stack_width, device=self.device
        )
        a_push, a_pop, a_no_op = controls[:, 0], controls[:, 1], controls[:, 2]
        stack_down = torch.cat((prev_stack[:, 1:], zeros_bottom), dim=1)
        stack_up = torch.cat((input_val, prev_stack[:, :-1]), dim=1)
        return a_no_op * prev_stack + a_push * stack_up + a_pop * stack_down

    def init_hidden(self):
        h = torch.zeros(
            self.n_layers * self.num_dir, 1, self.hidden_size, device=self.device
        )
        if self.has_cell:
            c = torch.zeros_like(h)
            return (h, c)
        return h

    def init_stack(self):
        if self.has_stack:
            return torch.zeros(
                1, self.stack_depth, self.stack_width, device=self.device
            )
        return None

    # --------------------------- training ---------------------------------
    def train_step(self, inp, target) -> float:
        self.train()
        hidden = self.init_hidden()
        stack = self.init_stack()
        self.optimizer.zero_grad()
        loss = torch.zeros(1, device=self.device)
        for c in range(len(inp)):
            output, hidden, stack = self.forward(inp[c], hidden, stack)
            loss = loss + self.criterion(output, target[c].unsqueeze(0))
        loss.backward()
        self.optimizer.step()
        return float(loss.item() / max(len(inp), 1))

    # --------------------------- sampling ---------------------------------
    def evaluate(
        self,
        data,
        prime_str: str = "<",
        end_token: str = ">",
        predict_len: Optional[int] = None,
        temperature: float = 1.0,
    ) -> str:
        """Sample a SMILES string from the model, primed with ``prime_str``.

        ``data`` should expose ``char_tensor``, ``all_characters``,
        ``start_token`` and ``max_len`` attributes (see ``GeneratorData``).
        """
        with torch.no_grad():
            self.eval()
            hidden = self.init_hidden()
            stack = self.init_stack() if self.has_stack else None

            if prime_str in ("^", "~"):
                full_prime = data.start_token + prime_str
            else:
                full_prime = prime_str
            prime_tensor = data.char_tensor(full_prime)
            sample = full_prime

            for p in range(len(prime_tensor) - 1):
                _, hidden, stack = self.forward(prime_tensor[p], hidden, stack)
            inp = prime_tensor[-1]

            limit = predict_len if predict_len is not None else data.max_len
            for _ in range(limit):
                output, hidden, stack = self.forward(inp, hidden, stack)
                dist = output.data.view(-1).div(temperature).exp()
                top_i = torch.multinomial(dist, 1)[0]
                ch = data.all_characters[top_i]
                sample += ch
                inp = data.char_tensor(ch)
                if ch == end_token:
                    break

            self.train()
            return sample

    def fit(
        self,
        data,
        n_iterations: int,
        all_losses: Optional[List[float]] = None,
        print_every: int = 5000,
        plot_every: int = 10,
        augment: bool = False,
    ) -> List[float]:
        """Maximum-likelihood training loop with optional SMILES augmentation."""
        if all_losses is None:
            all_losses = []
        start = time.time()
        loss_avg = 0.0

        smiles_aug = SmilesEnumerator() if augment else None

        pbar = trange(1, n_iterations + 1, desc="Training generator")
        for epoch in pbar:
            inp, target = data.random_training_set(smiles_aug)
            loss = self.train_step(inp, target)
            loss_avg += loss

            if epoch % print_every == 0:
                tqdm.write(
                    f"[{time_since(start)} ({epoch} {epoch / n_iterations * 100:.0f}%) {loss:.4f}]"
                )
                tqdm.write(
                    f"  ^ sample: {self.evaluate(data, prime_str='^')}"
                )
                tqdm.write(
                    f"  ~ sample: {self.evaluate(data, prime_str='~')}"
                )

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                pbar.set_postfix(loss=f"{loss_avg / plot_every:.4f}")
                loss_avg = 0.0

        return all_losses
