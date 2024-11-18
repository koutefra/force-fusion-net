from entities.vector2d import Velocity, Acceleration
from collections import OrderedDict
from entities.scene import Trajectories, Person

class FiniteDifferenceCalculator:
    def __init__(self, win_size: int, fdm_type: str = "backward"):
        self.win_size = win_size
        self.fdm_type = fdm_type
        if fdm_type != "backward":
            raise ValueError("Only backward FDM is currently supported.")
    
    def compute_velocities(self, trajectories: Trajectories, fps: float) -> None:
        for person_trajectory in trajectories.values():
            self._compute_fdm_property(person_trajectory, "position", "velocity", fps)

    def compute_accelerations(self, trajectories: Trajectories, fps: float) -> None:
        for person_trajectory in trajectories.values():
            self._compute_fdm_property(person_trajectory, "velocity", "acceleration", fps)

    def _compute_fdm_property(
        self,
        person_trajectory: OrderedDict[int, Person],
        in_prop: str,
        out_prop: str,
        fps: float
    ) -> None:
        frame_numbers = list(person_trajectory.keys())
        for frame_id, frame_number in enumerate(frame_numbers):
            window_frame_ids = list(range(max(0, frame_id - self.win_size + 1), frame_id + 1))
            
            window = [
                getattr(person_trajectory[frame_numbers[fid]], in_prop)
                for fid in window_frame_ids
                if frame_numbers[fid] in person_trajectory
            ]

            if len(window) == self.win_size:
                time_window = [
                    (frame_numbers[j] - frame_numbers[i]) / fps
                    for i, j in zip([id - 1 for id in window_frame_ids][1:], window_frame_ids[1:])
                ]
                computed_value = (
                    Velocity.from_points(window, time_window) if out_prop == "velocity"
                    else Acceleration.from_velocities(window, time_window)
                )
            
            setattr(person_trajectory[frame_number], out_prop, computed_value)
