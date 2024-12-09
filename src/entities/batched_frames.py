import torch
from itertools import islice
from entities.frame import Frames, Frame
import numpy as np

class BatchedFrames:
    def __init__(self, frames: list[Frames], person_ids: list[int], device: torch.device, dtype: torch.dtype):
        if len(frames) != len(person_ids):
            raise ValueError("Person ids must match to the frames.")
        if len(frames) == 0:
            raise ValueError("No data.")

        self.device = device
        self.dtype = dtype
        self.steps_count = len(frames[0])
        frames_by_step = [
            [next(islice(fs.values(), step, step + 1)) for fs in frames]
            for step in range(self.steps_count)
        ]

        self.person_positions, self.person_velocities, self.person_goals = self.batch_focus_persons(
            frames_by_step[0], 
            person_ids,
            device,
            dtype
        )

        self.other_positions_by_step, self.other_velocities_by_step, self.obstacles_by_step = [], [], []
        for frames in frames_by_step:
            other_positions, other_velocities = self.batch_other_persons(frames, person_ids, device, dtype)
            obstacles = self.batch_obstacles(frames, device, dtype)
            self.other_positions_by_step.append(other_positions)
            self.other_velocities_by_step.append(other_velocities)
            self.obstacles_by_step.append(obstacles)

        self.step = 0

    def update(self, new_person_positions: torch.Tensor, new_person_velocities: torch.Tensor) -> None:
        self.person_positions = new_person_positions
        self.person_velocities = new_person_velocities
        self.step = self.step + 1

    @staticmethod
    def batch_focus_persons(
        frames: list[Frame], 
        person_ids: list[int],
        device: torch.device,
        dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        person_positions = np.array([frame.persons[pid].position.to_numpy() for frame, pid in zip(frames, person_ids)])
        person_velocities = np.array([frame.persons[pid].velocity.to_numpy() for frame, pid in zip(frames, person_ids)])
        person_goals = np.array([frame.persons[pid].goal.to_numpy() for frame, pid in zip(frames, person_ids)])
        return (
            torch.tensor(person_positions, dtype=dtype, device=device),
            torch.tensor(person_velocities, dtype=dtype, device=device),
            torch.tensor(person_goals, dtype=dtype, device=device)
        )

    @staticmethod
    def batch_other_persons(
        frames: list[Frame], 
        person_ids: list[int], 
        device: torch.device,
        dtype: torch.dtype,
        padding_value: float = float("nan"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_others = max(len(frame.persons) - 1 for frame in frames)
        other_positions_all = []
        other_velocities_all = []
        for frame, person_id in zip(frames, person_ids):
            other_positions = [
                person.position.to_numpy()
                for pid, person in frame.persons.items()
                if pid != person_id and person.velocity
            ]
            other_velocities = [
                person.velocity.to_numpy()
                for pid, person in frame.persons.items()
                if pid != person_id and person.velocity
            ]

            # Pad positions and velocities to `max_others` using NumPy
            num_others = len(other_positions)
            if num_others < max_others:
                pad_size = (max_others - num_others, 2)
                other_positions = np.vstack([
                    np.array(other_positions),
                    np.full(pad_size, padding_value)
                ])
                other_velocities = np.vstack([
                    np.array(other_velocities),
                    np.full(pad_size, padding_value)
                ])
            else:
                other_positions = np.array(other_positions)
                other_velocities = np.array(other_velocities)
        
            other_positions_all.append(other_positions)
            other_velocities_all.append(other_velocities)
        return (
            torch.tensor(np.array(other_positions_all), dtype=dtype, device=device),
            torch.tensor(np.array(other_velocities_all), dtype=dtype, device=device)
        )

    @staticmethod
    def batch_obstacles(
        frames: list[Frame], 
        device: torch.device,
        dtype: torch.dtype,
        padding_value: float = float("nan")
    ) -> torch.Tensor:
        max_obstacles = max(len(frame.obstacles) - 1 for frame in frames)
        obstacle_positions_all = []
        for frame in frames:
            obstacle_positions = [
                obstacle.p1.to_list() + obstacle.p2.to_list()
                for obstacle in frame.obstacles.values()
            ]
            # Pad using NumPy to match `max_obstacles`
            num_obstacles = len(obstacle_positions)
            if num_obstacles < max_obstacles:
                pad_size = (max_obstacles - num_obstacles, len(obstacle_positions[0]))
                obstacle_positions = np.vstack([
                    np.array(obstacle_positions),
                    np.full(pad_size, padding_value)
                ])
            else:
                obstacle_positions = np.array(obstacle_positions)
            obstacle_positions_all.append(obstacle_positions)
        obstacle_positions_all = np.array(obstacle_positions_all)
        return torch.tensor(obstacle_positions_all, dtype=dtype, device=device)

    @staticmethod
    def compute_individual_features(
        person_positions: torch.Tensor,  # (batch_size, 2)
        person_velocities: torch.Tensor,  # (batch_size, 2)
        person_goals: torch.Tensor,  # (batch_size, 2)
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        diffs_to_goal = person_goals - person_positions
        dists_to_goal = torch.norm(diffs_to_goal, dim=1, keepdim=True)  # (batch_size, 1)
        directions = diffs_to_goal / (dists_to_goal + epsilon)  # (batch_size, 2)
        vel_towards_goal = torch.sum(person_velocities * directions, dim=1, keepdim=True)  # (batch_size, 1)
        features = torch.cat((
            person_velocities,  # Velocity components (batch_size, 2)
            dists_to_goal,      # Distance to goal (batch_size, 1)
            directions,         # Direction to goal components (batch_size, 2)
            vel_towards_goal    # Velocity towards goal (batch_size, 1)
        ), dim=1)  # Shape: (batch_size, 6)
        return features

    @staticmethod
    def compute_interaction_features(
        person_positions: torch.Tensor,  # (batch_size, 2)
        person_velocities: torch.Tensor,  # (batch_size, 2)
        other_positions: torch.Tensor,  # (batch_size, M, 2)
        other_velocities: torch.Tensor,  # (batch_size, M, 2)
        mask: torch.Tensor,  # (batch_size)
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1)
        
        diffs = other_positions - person_positions[:, None, :]  # (batch_size, M, 2)
        diffs = torch.where(mask_expanded, diffs, torch.zeros_like(diffs))
        distances = torch.norm(diffs, dim=2, keepdim=True)  # (batch_size, M, 1)
        distances = torch.where(mask_expanded, distances, torch.zeros_like(distances))

        directions = diffs / (distances + epsilon)  # (batch_size, M, 2)
        directions = torch.where(mask_expanded, directions, torch.zeros_like(directions))

        vel_diffs = other_velocities - person_velocities[:, None, :]
        vel_diffs = torch.where(mask_expanded, vel_diffs, torch.zeros_like(vel_diffs))
        relative_velocities = torch.norm(vel_diffs, dim=2, keepdim=True)  # (batch_size, M, 1)
        relative_velocities = torch.where(mask_expanded, relative_velocities, torch.zeros_like(relative_velocities))

        alignments = torch.sum(person_velocities[:, None, :] * directions, dim=2, keepdim=True)  # (batch_size, M, 1)
        alignments = torch.where(mask_expanded, alignments, torch.zeros_like(alignments))

        interaction_features = torch.cat((
            distances,         # Distance (batch_size, M, 1)
            directions,        # Direction components (batch_size, M, 2)
            relative_velocities,  # Relative velocity (batch_size, M, 1)
            alignments         # Alignment (batch_size, M, 1)
        ), dim=2)  # Shape: (batch_size, M, 5)
        return interaction_features

    @staticmethod
    def compute_obstacle_features(
        person_positions: torch.Tensor,  # (batch_size, 2)
        line_obstacles: torch.Tensor,  # (batch_size, n_line_obstacles, 4)
        mask: torch.Tensor,  # (batch_size, n_line_obstacles)
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        p1 = line_obstacles[:, :, :2]  # (batch_size, n_line_obstacles, 2)
        p2 = line_obstacles[:, :, 2:]  # (batch_size, n_line_obstacles, 2)
        p1_to_p2 = p2 - p1  # (batch_size, n_line_obstacles, 2)

        # Compute projections of the person onto the line segment
        p1_to_person = person_positions[:, None, :] - p1  # (batch_size, n_line_obstacles, 2)
        line_lengths_sq = torch.sum(p1_to_p2**2, dim=2, keepdim=True)  # (batch_size, n_line_obstacles, 1)
        projections = torch.sum(p1_to_person * p1_to_p2, dim=2, keepdim=True) / (line_lengths_sq + epsilon)
        projections = torch.clamp(projections, min=0, max=1)  # Clamp to segment [0, 1]
        closest_points = p1 + projections * p1_to_p2  # (batch_size, n_line_obstacles, 2)

        # Combine p1, p2, and closest_points into a single tensor
        all_points = torch.stack([closest_points, p1, p2], dim=2)  # (batch_size, n_line_obstacles, 3, 2)
        person_positions_expanded = person_positions[:, None, None, :]  # (batch_size, 1, 1, 2)
        diffs = all_points - person_positions_expanded  # (batch_size, n_line_obstacles, 3, 2)
        dists = torch.norm(diffs, dim=3, keepdim=True)  # (batch_size, n_line_obstacles, 3, 1)
        directions = diffs / (dists + epsilon)  # (batch_size, n_line_obstacles, 3, 2)

        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # (batch_size, n_line_obstacles, 1, 1)
        dists = torch.where(mask_expanded, dists, torch.zeros_like(dists))
        directions = torch.where(mask_expanded, directions, torch.zeros_like(directions))

        obstacle_features = torch.cat([dists, directions], dim=3)  # (batch_size, n_line_obstacles, 3, 3)
        # Reshape to final output shape: (batch_size, n_line_obstacles, 9)
        return obstacle_features.view(obstacle_features.size(0), obstacle_features.size(1), -1)

    def compute_all_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        individual_features = self.compute_individual_features(
            self.person_positions,
            self.person_velocities,
            self.person_goals
        )
        interaction_features = self.compute_interaction_features(
            self.person_positions,
            self.person_velocities,
            self.other_positions_by_step[self.step],
            self.other_velocities_by_step[self.step],
            mask=~torch.isnan(self.other_positions_by_step[self.step]).any(dim=-1)
        )
        obstacle_features = self.compute_obstacle_features(
            self.person_positions,
            self.obstacles_by_step[self.step],
            mask=~torch.isnan(self.obstacles_by_step[self.step]).any(dim=-1)
        )
        return individual_features, interaction_features, obstacle_features