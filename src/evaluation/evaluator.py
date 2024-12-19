from entities.scene import Scenes, Scene
from entities.frame import Frame, Frames, Trajectories, Trajectory
from models.base_model import BaseModel
from typing import Callable
import torch
from torch import Tensor
from collections import defaultdict
from entities.vector2d import Point2D
import math


class Evaluator:
    def __init__(self, bottleneck_range: tuple[Point2D, Point2D]):
        self.bottleneck_corner1 = bottleneck_range[0]
        self.bottleneck_corner2 = bottleneck_range[1]

    def get_evaluation_function(self, scenes: Scenes) -> Callable[[BaseModel, int, dict[str, torch.Tensor]], None]:
        def evaluate(model: BaseModel, epoch: int, logs: dict[str, torch.Tensor]) -> None:
            total_mae, total_ate, total_collision_obstacles = 0, 0, 0
            total_collision_persons, total_flow, total_density = 0, 0, 0
            total_jerk, total_velocity_consistency = 0, 0
            total_time_to_clear = 0

            scene_count = len(scenes)
            all_speeds = []
            true_positions, predicted_positions = [], []

            for scene in scenes:  # Iterate over all scenes
                scene_metrics = self._evaluate_scene(model, scene)

                # Aggregate metrics
                total_mae += scene_metrics['mae']
                total_ate += scene_metrics['ate']
                total_collision_obstacles += scene_metrics['collision_obstacles']
                total_collision_persons += scene_metrics['collision_persons']
                total_flow += scene_metrics['flow']
                total_density += scene_metrics['density']
                total_jerk += scene_metrics['jerk']
                total_velocity_consistency += scene_metrics['velocity_consistency']
                total_time_to_clear += scene_metrics['time_to_clear']

                all_speeds.extend(scene_metrics['speeds'])
                true_positions.extend(scene_metrics['true_positions'])
                predicted_positions.extend(scene_metrics['predicted_positions'])

            # Write to logs
            logs['mae'] = torch.tensor(total_mae / scene_count, device=model.device)
            logs['ate'] = torch.tensor(total_ate / scene_count, device=model.device)
            logs['collision_obstacles'] = torch.tensor(total_collision_obstacles, device=model.device)
            logs['collision_persons'] = torch.tensor(total_collision_persons, device=model.device)
            logs['flow'] = torch.tensor(total_flow / scene_count, device=model.device)
            logs['density'] = torch.tensor(total_density / scene_count, device=model.device)
            logs['jerk'] = torch.tensor(total_jerk / scene_count, device=model.device)
            logs['velocity_consistency'] = torch.tensor(total_velocity_consistency / scene_count, device=model.device)
            logs['time_to_clear'] = torch.tensor(total_time_to_clear / scene_count, device=model.device)

            # Statistical Metrics
            logs['kl_div'] = self._compute_kl_div(true_positions, predicted_positions, model.device)
            logs['speed_distribution'] = self._compare_speed_distributions(all_speeds, model.device)

        return evaluate

    def _evaluate_scene(self, ori_scene: Scene, sim_scene: Scene) -> dict:
        ori_trajectories = ori_scene.frames.to_trajectories()
        sim_trajectories = sim_scene.frames.to_trajectories()

        paired_trajectories = [
            (ori_trajectories[pid], sim_trajectories[pid])
            for pid in ori_trajectories.keys() & sim_trajectories.keys()
        ]

        for ori_trajectory, sim_trajectory in paired_trajectories:


        for person_id, ori_trajectory in ori_scene.frames.to_trajectories().items():
        for person_id, sim_trajectory in sim_scene.frames.to_trajectories().items():

            for frame_idx in range(len(scene.frames) - 1):
                frame = scene.frames[frame_idx]
                next_frame = scene.frames[frame_idx + 1]

                # Predict the next position
                predicted_position = model.predict(frame)
                true_position = next_frame.get_person_position(person)

                # Compute positional errors
                error = torch.norm(predicted_position - torch.tensor(true_position))
                mae += error.item()
                ate += error.item()

                # Record positions, velocities, and jerk
                true_positions.append(true_position)
                predicted_positions.append(predicted_position.tolist())
                velocity = torch.norm(predicted_position - torch.tensor(frame.get_person_position(person)))
                velocities.append(velocity)

                if len(velocities) > 1:
                    acceleration = velocities[-1] - velocities[-2]
                    accelerations.append(acceleration)

            # Compute jerk
            jerk += sum(abs(accelerations[i + 1] - accelerations[i]) for i in range(len(accelerations) - 1))

            # Compute flow and density
            if self._is_in_bottleneck(person.position):
                flow += 1
                speeds.append(velocity.item())

        # Aggregate metrics
        return {
            'mae': mae / len(scene.persons),
            'ate': ate / len(scene.persons),
            'collision_obstacles': collision_obstacles,
            'collision_persons': collision_persons,
            'flow': flow,
            'density': flow / self._bottleneck_area(),
            'jerk': jerk / len(scene.persons),
            'velocity_consistency': sum(velocities) / len(velocities) if velocities else 0,
            'time_to_clear': time_to_clear,
            'speeds': speeds,
            'true_positions': true_positions,
            'predicted_positions': predicted_positions,
        }

    def _is_in_bottleneck(self, position) -> bool:
        x1, y1, x2, y2 = self.bottleneck_coords
        return x1 <= position[0] <= x2 and y1 <= position[1] <= y2

    def _bottleneck_area(self) -> float:
        x1, y1, x2, y2 = self.bottleneck_coords
        return abs(x2 - x1) * abs(y2 - y1)

    def _compute_kl_div(self, true_positions, predicted_positions, device) -> Tensor:
        true_hist = torch.histc(torch.tensor(true_positions, device=device), bins=10)
        pred_hist = torch.histc(torch.tensor(predicted_positions, device=device), bins=10)
        kl_div = torch.nn.functional.kl_div(pred_hist.log(), true_hist, reduction="batchmean")
        return kl_div

    def _compare_speed_distributions(self, speeds, device) -> Tensor:
        speeds_tensor = torch.tensor(speeds, device=device)
        return speeds_tensor.mean(), speeds_tensor.std()
