import numpy as np
from itertools import islice
from entities.frame import Frames, Frame

class BatchedFrames:
    def __init__(self, frames: list[Frames], person_ids: list[int]):
        if len(frames) != len(person_ids):
            raise ValueError("Person ids must match to the frames.")

        if len(frames) == 0:
            raise ValueError("No data.")

        steps = len(frames[0])
        frames_by_step = [
            [next(islice(fs.items(), step, step + 1)) for fs in frames]
            for step in range(steps)
        ]

        self.person_positions, self.person_velocities, self.person_goals = self.batch_focus_persons(
            frames_by_step[0], 
            person_ids
        )

        self.other_positions_by_step, self.other_velocities_by_step = [], [] 
        for frames in frames_by_step:
            other_positions, other_velocities = self.batch_other_persons(frames, person_ids)
            self.other_positions_by_step.append(other_positions)
            self.other_velocities_by_step.append(other_velocities)

        self.step = 0

    def update(self, new_person_positions: np.ndarray, new_person_velocities: np.ndarray) -> None:
        self.person_positions = new_person_positions
        self.person_velocities = new_person_velocities
        self.step = self.step + 1

    @staticmethod
    def batch_focus_persons(
        frames: list[Frame], 
        person_ids: list[int]
    ) -> tuple[np.array, np.array, np.array]:
        person_positions = []
        person_velocities = []
        person_goals = []
        for frame, person_id in zip(frames, person_ids):
            person = frame.persons[person_id]
            person_positions.append(person.position.to_numpy())
            person_velocities.append(person.velocity.to_numpy())
            person_goals.append(person.goal.to_numpy())
        return (
            np.array(person_positions), 
            np.array(person_velocities), 
            np.array(person_goals)
        )

    @staticmethod
    def batch_other_persons(
        frames: list[Frame], 
        person_ids: list[int], 
        padding_value: float = np.nan
    ) -> tuple[np.array, np.array]:
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
                other_positions.append(np.array([padding_value, padding_value]))
                other_velocities.append(np.array([padding_value, padding_value]))

            other_positions_all.append(other_positions)
            other_velocities_all.append(other_velocities)

        other_positions_all = np.array(other_positions_all)
        other_velocities_all = np.array(other_velocities_all)

        return other_positions_all, other_velocities_all

    @staticmethod
    def compute_individual_features(
        person_positions: np.ndarray,  # (batch_size, 2)
        person_velocities: np.ndarray,  # (batch_size, 2)
        person_goals: np.ndarray,  # (batch_size, 2)
        epsilon: float = 1e-8
    ) -> np.ndarray:
        diffs_to_goal = person_goals - person_positions
        dists_to_goal = np.linalg.norm(diffs_to_goal, axis=1, keepdims=True)  # (batch_size, 1)
        # Avoid division by zero by replacing zero distances with a small epsilon
        dists_to_goal = np.maximum(dists_to_goal, epsilon)
        directions = diffs_to_goal / dists_to_goal  # (batch_size, 2)
        vel_towards_goal = np.sum(person_velocities * directions, axis=1, keepdims=True)  # (batch_size, 1)

        features = np.hstack((
            person_velocities,          # Velocity components (batch_size, 2)
            dists_to_goal,              # Distance to goal (batch_size, 1)
            directions,                 # Direction to goal components (batch_size, 2)
            vel_towards_goal            # Velocity towards goal (batch_size, 1)
        ))
        return features  # Shape: (batch_size, 6)

    @staticmethod
    def compute_interaction_features(
        person_positions: np.ndarray,  # (batch_size, 2)
        person_velocities: np.ndarray,  # (batch_size, 2)
        other_positions: np.ndarray,  # (batch_size, M, 2)
        other_velocities: np.ndarray,  # (batch_size, M, 2)
        epsilon: float = 1e-8
    ) -> np.ndarray:
        diffs = other_positions - person_positions[:, None, :]  # (batch_size, M, 2)
        distances = np.linalg.norm(diffs, axis=2, keepdims=True)  # (batch_size, M, 1)
        distances = np.maximum(distances, epsilon)  # Avoid division by zero
        directions = diffs / distances  # (batch_size, M, 2)
        relative_velocities = np.linalg.norm(other_velocities - person_velocities[:, None, :], axis=2, keepdims=True)  # (batch_size, M, 1)
        # Compute alignment (dot product of velocity with direction vector)
        alignments = np.sum(person_velocities[:, None, :] * directions, axis=2, keepdims=True)  # (batch_size, M, 1)

        interaction_features = np.concatenate((
            distances,         # Distance (batch_size, M, 1)
            directions,        # Direction components (batch_size, M, 2)
            relative_velocities,  # Relative velocity (batch_size, M, 1)
            alignments         # Alignment (batch_size, M, 1)
        ), axis=2)  # (batch_size, M, 5)
        return interaction_features

    @staticmethod
    def compute_obstacle_features(
        person_positions: np.ndarray,  # (batch_size, 2)
        line_obstacles: np.ndarray,  # (n_line_obstacles, 4) Each row: [x1, y1, x2, y2]
        epsilon: float = 1e-8
    ) -> np.ndarray:
        # Extract obstacle endpoints
        p1 = line_obstacles[:, :2]  # (n_line_obstacles, 2)
        p2 = line_obstacles[:, 2:]  # (n_line_obstacles, 2)
        # Vectorized closest point calculation
        p1_to_p2 = p2 - p1  # (n_line_obstacles, 2), direction vectors for obstacles
        p1_to_person = person_positions[:, None, :] - p1[None, :, :]  # (batch_size, n_line_obstacles, 2)
        # Compute projections for closest point
        line_lengths_sq = np.sum(p1_to_p2**2, axis=1, keepdims=True)  # (n_line_obstacles, 1)
        projections = np.sum(p1_to_person * p1_to_p2[None, :, :], axis=2) / np.maximum(line_lengths_sq.T, epsilon)  # (batch_size, n_line_obstacles)
        projections = np.clip(projections, 0, 1)  # Clamp to segment
        # Compute closest points on lines
        closest_points = p1[None, :, :] + projections[:, :, None] * p1_to_p2[None, :, :]  # (batch_size, n_line_obstacles, 2)

        # Loop through each point type: closest, start, end
        obstacle_features = []
        for point_type, point_positions in zip(
            ["closest", "start", "end"],
            [closest_points, p1[None, :, :], p2[None, :, :]]
        ):
            diffs = point_positions - person_positions[:, None, :]  # (batch_size, n_line_obstacles, 2)
            dists = np.linalg.norm(diffs, axis=2, keepdims=True)  # (batch_size, n_line_obstacles, 1)
            dists = np.maximum(dists, epsilon)  # Avoid division by zero
            directions = diffs / dists  # (batch_size, n_line_obstacles, 2)
            obstacle_features.append(np.concatenate([dists, directions], axis=2))  # (batch_size, n_line_obstacles, 3)

        obstacle_features = np.concatenate(obstacle_features, axis=2)  # (batch_size, n_line_obstacles, 9)
        return obstacle_features

    def compute_all_features(
        self,
        obstacles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            obstacles
        )
        return individual_features, interaction_features, obstacle_features