import torch
import torch.nn as nn
import torch.nn.functional as F
from entities.vector2d import kinematic_equation
from models.trainable_module import TrainableModule

class NeuralNetModel(TrainableModule):
    def __init__(
        self, 
        individual_fts_dim: int, 
        interaction_fts_dim: int, 
        obstacle_fts_dim: int, 
        hidden_dims: list[int] 
    ):
        super(NeuralNetModel, self).__init__()
        self.individual_fts_dim = individual_fts_dim
        self.interaction_fts_dim = interaction_fts_dim
        self.obstacle_fts_dim = obstacle_fts_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 2

        self.fc_interaction = nn.Linear(interaction_fts_dim, hidden_dims[0])
        self.fc_obstacle = nn.Linear(obstacle_fts_dim, hidden_dims[0])

        # make for hidden_dims of fcs
        combined_input_dim = individual_fts_dim + 2 * hidden_dims[-1]
        self.fcs_combined = self._build_mlp(combined_input_dim, hidden_dims[1:])
        self.output_layer = nn.Linear(hidden_dims[-1], self.output_dim)

    def forward(
        self, 
        x_individual: torch.Tensor, 
        x_interaction: torch.Tensor,
        x_obstacle: torch.Tensor
    ) -> torch.Tensor:
        # x_individual.shape: [batch_size, individual_fts_dim] 
        # x_interaction.shape: [batch_size, l, interaction_fts_dim]
        # x_obstacle.shape: [batch_size, k, obstacle_fts_dim]

        def process_var_length_features(x, fc_layer):
            if x.numel() > 0:
                orig_shape = x.shape
                x = x.view(-1, x.size(-1))
                x = F.relu(fc_layer(x))
                return x.view(*orig_shape[:-1], -1).sum(dim=1)
            else:
                return torch.zeros(x_individual.size(0), fc_layer.out_features, device=x.device, dtype=x.dtype)

        x_interaction = process_var_length_features(x_interaction, self.fc_interaction)
        x_obstacle = process_var_length_features(x_obstacle, self.fc_obstacle)

        combined_features = torch.cat([x_individual, x_interaction, x_obstacle], dim=-1)
        output = self.fcs_combined(combined_features)
        output = self.output_layer(output)
        return output

    def predict(self, dataloader, as_numpy=True):
        """Compute predictions for the given dataset. Overriden method.

        - `dataloader` is the dataset to predict on, each element either
          directly the input or a tuple whose first element is the input;
          the input can be either a single tensor or a tuple of tensors;
        - `as_numpy` is a flag controlling whether the output should be
          converted to a numpy array or kept as a PyTorch tensor.

        The method returns a Python list whose elements are predictions
        of the individual examples. Note that if the input was padded, so
        will be the predictions, which will then need to be trimmed."""
        self.eval()
        predictions = []
        for batched_frames, _ in dataloader:
            xs = batched_frames.compute_all_features()
            xs = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,)))
            predictions.extend(self.predict_step(xs, as_numpy=as_numpy))
        return predictions

    def fit(self, dataloader, epochs, dev=None, callbacks=[], verbose=1):
        """Train the model on the given dataset. Overriden method.

        - `dataloader` is the training dataset, each element a pair of inputs and an output;
          the inputs can be either a single tensor or a tuple of tensors;
        - `dev` is an optional development dataset;
        - `epochs` is the number of epochs to train;
        - `callbacks` is a list of callbacks to call after each epoch with
          arguments `self`, `epoch`, and `logs`;
        - `verbose` controls the verbosity: 0 for silent, 1 for persistent progress bar,
          2 for a progress bar only when writing to a console.
        """
        logs = {}
        for epoch in range(epochs):
            self.train()
            self.loss_metric.reset()
            self.metrics.reset()
            start = self._time()
            epoch_message = f"Epoch={epoch+1}/{epochs}"
            data_and_progress = self._tqdm(
                dataloader, epoch_message, unit="batch", leave=False, disable=None if verbose == 2 else not verbose)
            for batched_frames, ys in data_and_progress:
                ys = tuple(y.to(self.device) for y in (ys if isinstance(ys, tuple) else (ys,)))
                gts, delta_times = ys
                self.zero_grad()
                for _ in range(batched_frames.steps_count):
                    xs = batched_frames.compute_all_features()
                    xs = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,)))
                    pred_accs = self.forward(*xs)
                    new_pos, new_vel = self.apply_predictions(
                        cur_positions=batched_frames.person_positions,
                        cur_velocities=batched_frames.person_velocities,
                        delta_times=delta_times,
                        pred_accs=pred_accs
                    )
                    batched_frames.update(new_pos, new_vel)

                logs = self.backprop(pred_accs, gts, *xs)
                message = [epoch_message] + [f"{k}={v:#.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()]
                data_and_progress.set_description(" ".join(message), refresh=False)
            if dev is not None:
                logs |= {"dev_" + k: v for k, v in self.evaluate(dev, verbose=0).items()}
            for callback in callbacks:
                callback(self, epoch, logs)
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("dev_")}, epoch + 1)
            self.add_logs("dev", {k[4:]: v for k, v in logs.items() if k.startswith("dev_")}, epoch + 1)
            verbose and print(epoch_message, "{:.1f}s".format(self._time() - start),
                              *[f"{k}={v:#.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()])
        return logs

    def backprop(self, y_pred, ys, *xs):
        """An overridable method performing a single training step.

        A dictionary with the loss and metrics should be returned."""
        loss = self.compute_loss(y_pred, ys, *xs)
        loss.backward()
        with torch.no_grad():
            self.optimizer.step()
            self.schedule is not None and self.schedule.step()
            self.loss_metric.update(loss)
            return {"loss": self.loss_metric.compute()} \
                | ({"lr": self.schedule.get_last_lr()[0]} if self.schedule else {}) \
                | self.compute_metrics(y_pred, ys, *xs, training=True)

    def apply_predictions(self, cur_positions, cur_velocities, delta_times, pred_accs) -> tuple[torch.Tensor, torch.Tensor]:
        next_pos_predicted, next_vel_predicted = kinematic_equation(
            cur_positions=cur_positions,
            cur_velocities=cur_velocities,
            delta_times=delta_times,
            cur_accelerations=pred_accs
        )
        return next_pos_predicted, next_vel_predicted

    @staticmethod
    def from_weight_file(path: str, device: str | torch.device = "cpu") -> "NeuralNetModel":
        state_dict = torch.load(path, map_location=device)

        # Infer the dimensions from the loaded state dict
        individual_fts_dim = state_dict['fc_individual.weight'].size(1)
        interaction_fts_dim = state_dict['fc_interaction.weight'].size(1)
        obstacle_fts_dim = state_dict['fc_obstacle.weight'].size(1)
        hidden_dim = state_dict['fc_individual.weight'].size(0)

        # Create the model with inferred dimensions
        model = NeuralNetModel(
            individual_fts_dim=individual_fts_dim,
            interaction_fts_dim=interaction_fts_dim,
            obstacle_fts_dim=obstacle_fts_dim,
            hidden_dim=hidden_dim
        )

        # Load weights
        model.load_weights(path, device)
        return model

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _process_feature(self, x: torch.Tensor, layer: nn.Linear, out_dim: int) -> torch.Tensor:
        if x.numel() > 0:
            x = F.relu(layer(x))
            return torch.sum(x, dim=1)
        else:
            return torch.zeros(x.shape[0], out_dim, device=self.device)