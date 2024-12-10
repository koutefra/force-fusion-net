from entities.vector2d import Velocity, Acceleration
from collections import OrderedDict
from entities.frame import Trajectories, Trajectory
from entities.person import Person

class FiniteDifferenceCalculator:
    def __init__(self, win_size: int, fdm_type: str = "backward"):
        self.win_size = win_size
        self.fdm_type = fdm_type
        if fdm_type != "backward":
            raise ValueError("Only backward FDM is currently supported.")
    
    def compute_velocities(self, trajectories: Trajectories, fps: float) -> Trajectories:
        return self._compute_property(trajectories, "position", "velocity", fps)

    def compute_accelerations(self, trajectories: Trajectories, fps: float) -> Trajectories:
        return self._compute_property(trajectories, "velocity", "acceleration", fps)

    def _compute_property(
        self, 
        trajectories: Trajectories, 
        in_prop: str, 
        out_prop: str, 
        fps: float
    ) -> Trajectories:
        new_trajectories_data = {}
        for person_id, person_trajectory in trajectories.items():
            computed_properties = self._compute_fdm_property(person_trajectory, in_prop, out_prop, fps)
            new_records = OrderedDict()
            for frame_number, computed_value in computed_properties.items():
                ori_person = person_trajectory.records[frame_number]
                new_person = Person(
                    id=person_id,
                    position=ori_person.position,
                    goal=ori_person.goal,
                    velocity=ori_person.velocity if out_prop != "velocity" else computed_value,
                    acceleration=ori_person.acceleration if out_prop != "acceleration" else computed_value
                )
                new_records[frame_number] = new_person
            new_trajectories_data[person_id] = Trajectory(person_id=person_id, records=new_records)
        return Trajectories(new_trajectories_data)

    def _compute_fdm_property(
        self,
        person_trajectory: Trajectory,
        in_prop: str,
        out_prop: str,
        fps: float
    ) -> OrderedDict[int, Velocity | Acceleration]:
        computed_results = OrderedDict()
        frame_numbers = list(person_trajectory.records.keys())
        for frame_id, frame_number in enumerate(frame_numbers):
            window_frame_ids = list(range(max(0, frame_id - self.win_size + 1), frame_id + 1))
            
            window = [
                getattr(person_trajectory.records[frame_numbers[fid]], in_prop)
                for fid in window_frame_ids
                if frame_numbers[fid] in person_trajectory.records
                and
                getattr(person_trajectory.records[frame_numbers[fid]], in_prop) is not None
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
            else:
                computed_value = None
            computed_results[frame_number] = computed_value  # Store the computed result in a new dict
        return computed_results