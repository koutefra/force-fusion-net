import torch
from itertools import islice
from entities.frame import Frames, Frame
import numpy as np

class BatchedFrames:
    def __init__(
        self, 
        frames: list[Frames], 
        person_ids: list[int], 
        device: torch.device, 
        epsilon: float = 1e-8,
        dtype: torch.dtype = torch.float32
    ):
        if len(frames) != len(person_ids):
            raise ValueError("Person ids must match to the frames.")
        if len(frames) == 0:
            raise ValueError("No data.")

        self.device = device
        self.epsilon = epsilon
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

        self.other_positions_by_step = []
        self.other_velocities_by_step = []
        self.other_mask_by_step = []
        self.obstacles_by_step = []
        self.obstacle_mask_by_step = []
        for frames in frames_by_step:
            other_positions, other_velocities, other_mask = self.batch_other_persons(frames, person_ids, device, dtype)
            obstacles, obstacle_mask = self.batch_obstacles(frames, device, dtype)
            self.other_positions_by_step.append(other_positions)
            self.other_velocities_by_step.append(other_velocities)
            self.other_mask_by_step.append(other_mask)
            self.obstacles_by_step.append(obstacles)
            self.obstacle_mask_by_step.append(obstacle_mask)

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
        dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_others = max(1, max(len(frame.persons) - 1 for frame in frames))
        other_positions_all = []
        other_velocities_all = []
        mask_all = []

        for frame, person_id in zip(frames, person_ids):
            valid_other_persons = [person for pid, person in frame.persons.items()
                                if pid != person_id and person.velocity]

            num_others = len(valid_other_persons)
            other_positions = np.full((max_others, 2), 0.0, dtype=np.float32)
            other_velocities = np.full((max_others, 2), 0.0, dtype=np.float32)
            mask = np.full((max_others,), 0.0, dtype=bool)

            if num_others > 0:
                valid_positions = np.array([p.position.to_numpy() for p in valid_other_persons], dtype=np.float32)
                valid_velocities = np.array([p.velocity.to_numpy() for p in valid_other_persons], dtype=np.float32)

                other_positions[:num_others, :] = valid_positions
                other_velocities[:num_others, :] = valid_velocities
                mask[:num_others] = True

            other_positions_all.append(other_positions)
            other_velocities_all.append(other_velocities)
            mask_all.append(mask)

        return (
            torch.tensor(np.array(other_positions_all), dtype=dtype, device=device),
            torch.tensor(np.array(other_velocities_all), dtype=dtype, device=device),
            torch.tensor(np.array(mask_all), dtype=torch.bool, device=device)
        )

    @staticmethod
    def batch_obstacles(
        frames: list[Frame], 
        device: torch.device,
        dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_obstacles = max(1, max(len(frame.obstacles) for frame in frames))
        obstacle_positions_all = []
        mask_all = []
        for frame in frames:
            obstacle_positions = [
                obstacle.p1.to_list() + obstacle.p2.to_list()
                for obstacle in frame.obstacles.values()
            ]
            # Pad using NumPy to match `max_obstacles`
            num_obstacles = len(obstacle_positions)
            mask = np.zeros(max_obstacles, dtype=bool)
            if num_obstacles > 0:
                obstacle_positions = np.array(obstacle_positions, dtype=np.float32)
                mask[:num_obstacles] = True  # Update mask to True where data is valid
                if num_obstacles < max_obstacles:
                    pad_size = (max_obstacles - num_obstacles, len(obstacle_positions[0]))
                    obstacle_positions = np.vstack([
                        obstacle_positions,
                        np.full(pad_size, 0.0, dtype=np.float32)
                    ])
            else:
                obstacle_positions = np.full((max_obstacles, 4), 0.0, dtype=np.float32)
            obstacle_positions_all.append(obstacle_positions)
            mask_all.append(mask)
        return (
            torch.tensor(np.array(obstacle_positions_all), dtype=dtype, device=device),
            torch.tensor(np.array(mask_all), dtype=torch.bool, device=device)
        )

    @staticmethod
    def compute_individual_features(
        person_positions: torch.Tensor,  # (batch_size, 2)
        person_velocities: torch.Tensor,  # (batch_size, 2)
        person_goals: torch.Tensor,  # (batch_size, 2)
        epsilon: float
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
    def get_individual_feature_mapping():
        feature_names = ['vel_x', 'vel_y', 'goal_dist', 'goal_dir_x', 'goal_dir_y', 'goal_vel']
        return {name: index for index, name in enumerate(feature_names)}

    @staticmethod
    def get_individual_feature_index(name: str):
        return BatchedFrames.get_individual_feature_mapping()[name]

    @staticmethod
    def compute_interaction_features(
        person_positions: torch.Tensor,  # (batch_size, 2)
        person_velocities: torch.Tensor,  # (batch_size, 2)
        other_positions: torch.Tensor,  # (batch_size, M, 2)
        other_velocities: torch.Tensor,  # (batch_size, M, 2)
        epsilon: float
    ) -> torch.Tensor:
        diffs = other_positions - person_positions[:, None, :]  # (batch_size, M, 2)
        distances = torch.norm(diffs, dim=2, keepdim=True)  # (batch_size, M, 1)
        directions = diffs / (distances + epsilon)  # (batch_size, M, 2)
        vel_diffs = other_velocities - person_velocities[:, None, :]
        relative_velocities = torch.norm(vel_diffs, dim=2, keepdim=True)  # (batch_size, M, 1)
        alignments = torch.sum(person_velocities[:, None, :] * directions, dim=2, keepdim=True)  # (batch_size, M, 1)
        interaction_features = torch.cat((
            distances,         # Distance (batch_size, M, 1)
            directions,        # Direction components (batch_size, M, 2)
            relative_velocities,  # Relative velocity (batch_size, M, 1)
            alignments         # Alignment (batch_size, M, 1)
        ), dim=2)  # Shape: (batch_size, M, 5)
        return interaction_features

    @staticmethod
    def get_interaction_feature_mapping():
        feature_names = ['dist', 'dir_x', 'dir_y', 'rel_vel', 'alig']
        return {name: index for index, name in enumerate(feature_names)}

    @staticmethod
    def get_interaction_feature_index(name: str):
        return BatchedFrames.get_interaction_feature_mapping()[name]

    @staticmethod
    def compute_obstacle_features(
        person_positions: torch.Tensor,  # (batch_size, 2)
        line_obstacles: torch.Tensor,  # (batch_size, n_line_obstacles, 4)
        epsilon: float
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

        obstacle_features = torch.cat([dists, directions], dim=3)  # (batch_size, n_line_obstacles, 3, 3)
        # Reshape to final output shape: (batch_size, n_line_obstacles, 9)
        return obstacle_features.view(obstacle_features.size(0), obstacle_features.size(1), -1)

    @staticmethod
    def get_obstacle_feature_mapping():
        feature_names = [
            'dist_closest', 'dir_closest_x', 'dir_closest_y', 
            'dist_p1', 'dir_p1_x', 'dir_p1_y',
            'dist_p2', 'dir_p2_x', 'dir_p2_y'
        ]
        return {name: index for index, name in enumerate(feature_names)}

    @staticmethod
    def get_obstacle_feature_index(name: str):
        return BatchedFrames.get_obstacle_feature_mapping()[name]

    def compute_all_features(
        self
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns preprocessed tensor features. 
        
        The first tensor in the tuple is reserved for the individual features. The second represents
        the interaction features together with its mask. Third, also with mask, corresponds to the
        obstacle features.
        """
        individual_features = self.compute_individual_features(
            self.person_positions,
            self.person_velocities,
            self.person_goals,
            self.epsilon
        )
        interaction_features = self.compute_interaction_features(
            self.person_positions,
            self.person_velocities,
            self.other_positions_by_step[self.step],
            self.other_velocities_by_step[self.step],
            self.epsilon
        )
        obstacle_features = self.compute_obstacle_features(
            self.person_positions,
            self.obstacles_by_step[self.step],
            self.epsilon
        )
        interaction = (interaction_features, self.other_mask_by_step[self.step])
        obstacle = (obstacle_features, self.obstacle_mask_by_step[self.step])
        return individual_features, interaction, obstacle