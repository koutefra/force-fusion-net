import torch
from itertools import islice
from entities.frame import Frames, Frame

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
            [next(islice(fs.items(), step, step + 1)) for fs in frames]
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

    def batch_focus_persons(
        frames: list[Frame], 
        person_ids: list[int],
        device: torch.device,
        dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        person_positions = []
        person_velocities = []
        person_goals = []
        for frame, person_id in zip(frames, person_ids):
            person = frame.persons[person_id]
            person_positions.append(person.position.to_numpy())
            person_velocities.append(person.velocity.to_numpy())
            person_goals.append(person.goal.to_numpy())
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
            other_positions = []
            other_velocities = []
            for other_person_id, other_person in frame.persons.items():
                if other_person_id != person_id:
                    other_positions.append(other_person.position.to_numpy())
                    other_velocities.append(other_person.velocity.to_numpy())

            # Pad the arrays to match `max_others`
            while len(other_positions) < max_others:
                other_positions.append([padding_value, padding_value])
                other_velocities.append([padding_value, padding_value])
            other_positions_all.append(other_positions)
            other_velocities_all.append(other_velocities)
        return (
            torch.tensor(other_positions_all, dtype=dtype, device=device),
            torch.tensor(other_velocities_all, dtype=dtype, device=device)
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
            obstacle_positions = []
            for obstacle in frame.obstacles.values():
                points = obstacle.p1.to_list() + obstacle.p2.to_list()
                obstacle_positions.append(points)
            # Pad the arrays to match `max_obstacles`
            while len(obstacle_positions) < max_obstacles:
               obstacle_positions.append([padding_value, padding_value])
            obstacle_positions_all.append(obstacle_positions)
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
        dists_to_goal = torch.clamp(dists_to_goal, min=epsilon)  # Avoid division by zero
        directions = diffs_to_goal / dists_to_goal  # (batch_size, 2)
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
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        diffs = other_positions - person_positions[:, None, :]  # (batch_size, M, 2)
        distances = torch.norm(diffs, dim=2, keepdim=True)  # (batch_size, M, 1)
        distances = torch.clamp(distances, min=epsilon)  # Avoid division by zero
        directions = diffs / distances  # (batch_size, M, 2)
        relative_velocities = torch.norm(other_velocities - person_velocities[:, None, :], dim=2, keepdim=True)  # (batch_size, M, 1)
        alignments = torch.sum(person_velocities[:, None, :] * directions, dim=2, keepdim=True)  # (batch_size, M, 1)

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
        line_obstacles: torch.Tensor,  # (n_line_obstacles, 4)
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        p1 = line_obstacles[:, :2]  # (n_line_obstacles, 2)
        p2 = line_obstacles[:, 2:]  # (n_line_obstacles, 2)
        p1_to_p2 = p2 - p1  # (n_line_obstacles, 2)
        p1_to_person = person_positions[:, None, :] - p1[None, :, :]  # (batch_size, n_line_obstacles, 2)
        line_lengths_sq = torch.sum(p1_to_p2**2, dim=1, keepdim=True)  # (n_line_obstacles, 1)
        projections = torch.sum(p1_to_person * p1_to_p2[None, :, :], dim=2) / torch.clamp(line_lengths_sq.T, min=epsilon)  # (batch_size, n_line_obstacles)
        projections = torch.clamp(projections, min=0, max=1)  # Clamp to segment
        closest_points = p1[None, :, :] + projections[:, :, None] * p1_to_p2[None, :, :]  # (batch_size, n_line_obstacles, 2)

        obstacle_features = []
        for point_positions in [closest_points, p1[None, :, :], p2[None, :, :]]:
            diffs = point_positions - person_positions[:, None, :]  # (batch_size, n_line_obstacles, 2)
            dists = torch.norm(diffs, dim=2, keepdim=True)  # (batch_size, n_line_obstacles, 1)
            dists = torch.clamp(dists, min=epsilon)  # Avoid division by zero
            directions = diffs / dists  # (batch_size, n_line_obstacles, 2)
            obstacle_features.append(torch.cat([dists, directions], dim=2))  # (batch_size, n_line_obstacles, 3)

        return torch.cat(obstacle_features, dim=2)  # (batch_size, n_line_obstacles, 9)

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
            self.other_velocities_by_step[self.step]
        )
        obstacle_features = self.compute_obstacle_features(
            self.person_positions,
            self.obstacles_by_step[self.step]
        )
        return individual_features, interaction_features, obstacle_features