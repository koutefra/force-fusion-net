import math
from dataclasses import dataclass
from entities.person import Person
from entities.obstacle import LineObstacle
from collections import OrderedDict, defaultdict
from typing import Callable, Any
from entities.vector2d import Point2D, Velocity, Acceleration, closest_point_on_line
from itertools import islice
from tqdm import tqdm

@dataclass(frozen=True)
class Frame:
    number: int 
    persons: dict[int, Person]
    obstacles: list[LineObstacle]

    def normalized(
        self, 
        pos_scale: Callable[[Point2D], Point2D] = lambda f: f, 
        vel_scale: Callable[[Velocity], Velocity] = lambda v: v, 
        acc_scale: Callable[[Acceleration], Acceleration] = lambda a: a) -> "Frame":
        return Frame(
            number=self.number,
            persons={
                person_id: person.normalized(pos_scale, vel_scale, acc_scale)
                for person_id, person in self.persons.items()
            },
            obstacles=[o.normalized(pos_scale, vel_scale, acc_scale) for o in self.obstacles]
        )

    def get_person_ids_outside(self, bounding_box: tuple[float, float]) -> list[int]:
        return [
            pid 
            for pid, person in self.persons.items() 
            if not person.position.is_within(bounding_box[0], bounding_box[1])
        ]

    def get_person_ids_at_goal(self, goal_radius: float) -> list[int]:
        return [
            pid 
            for pid, person in self.persons.items() 
            if person.position.is_within(person.goal - goal_radius, person.goal + goal_radius)
        ]

    def remove_persons(self, person_ids: list[int]) -> "Frame":
        return Frame(
            number=self.number,
            persons={pid: person for pid, person in self.persons.items() if pid not in person_ids},
            obstacles=self.obstacles
        )

    def apply_kinematic_equation(self, delta_time: float, frame_step: int) -> "Frame":
        return Frame(
            number=self.number + frame_step,
            persons={
                pid: person.apply_kinematic_equation(delta_time) 
                for pid, person in self.persons.items()
                if person.velocity and person.acceleration
            },
            obstacles=self.obstacles
        )

    def add_persons(self, persons: dict[int, Person]) -> "Frame":
        new_persons = {**persons, **self.persons}
        return Frame(
            number=self.number,
            persons=new_persons,
            obstacles=self.obstacles
        )

    def set_accelerations(self, accs: dict[int, Acceleration]) -> "Frame":
        persons_to_update = {}
        persons_to_keep = {}

        for pid, person in self.persons.items():
            if pid in accs:
                persons_to_update[pid] = person.set_acceleration(accs[pid])
            else:
                persons_to_keep[pid] = person

        new_persons = {**persons_to_keep, **persons_to_update}
        return Frame(
            number=self.number,
            persons=new_persons,
            obstacles=self.obstacles
        )

    def get_all_features(
        self, 
        person_id: int
    ) -> tuple[list[float], list[list[float]], list[list[float]]]:
        person = self.persons[person_id]
        individual_features = person.get_individual_features()
        interaction_features = self.get_interaction_features(person)
        obstacle_features = self.get_obstacle_features(person)
        return individual_features, interaction_features, obstacle_features

    def get_interaction_features(self, person: Person) -> list[list[float]]:
        interaction_features = []
        for other_person_id, other_person in self.persons.items():
            if person.id == other_person_id or other_person.velocity is None:
                continue
            
            distance = (person.position - other_person.position).magnitude()
            direction_vector = person.position.direction_to(other_person.position)
            relative_velocity = (person.velocity - other_person.velocity).magnitude()
            alignment = person.velocity.dot(direction_vector)
            
            interaction_features.append([
                distance,
                direction_vector.x,
                direction_vector.y,
                relative_velocity,
                alignment,
            ])
        return interaction_features

    def get_obstacle_features(self, person: Person) -> list[list[float]]:
        obstacle_features = []
        for line in self.obstacles:
            closest_point = closest_point_on_line(person.position, line.p1, line.p2)

            distances, directions = {}, {}
            for name, point in [('closest', closest_point), ('start', line.p1), ('end', line.p2)]:
                distances[name] = (person.position - point).magnitude()
                directions[name] = person.position.direction_to(point)

            obstacle_features.append([
                distances['closest'],
                directions['closest'].x,
                directions['closest'].y,
                distances['start'],
                directions['start'].x,
                directions['start'].y,
                distances['end'],
                directions['end'].x,
                directions['end'].y
            ])
        return obstacle_features

class Frames(OrderedDict[int, Frame]):
    def normalized(
        self, 
        pos_scale: Callable[[Point2D], Point2D] = lambda f: f, 
        vel_scale: Callable[[Velocity], Velocity] = lambda v: v, 
        acc_scale: Callable[[Acceleration], Acceleration] = lambda a: a) -> "Frames":
        return Frames(OrderedDict({
            frame_number: frame.normalized(pos_scale, vel_scale, acc_scale)
            for frame_number, frame in self.items()
        }))

    def approximate_velocities(
        self,
        n_window_elements: int, 
        frame_step: int, 
        delta_time: float,
        fdm_method: str,  # "backward" or "central"
        print_progress: bool = True
    ) -> "Frames":
        return Frames.from_trajectories(
            Trajectories.from_frames(self).approximate_velocities(n_window_elements, frame_step, delta_time, fdm_method, print_progress),
            obstacles={frame_number: frame.obstacles for frame_number, frame in self.items()}
        )

    def approximate_accelerations(
        self,
        n_window_elements: int, 
        frame_step: int, 
        delta_time: float,
        fdm_method: str,  # "backward" or "central"
        print_progress: bool = True
    ) -> "Frames":
        return Frames.from_trajectories(
            Trajectories.from_frames(self).approximate_accelerations(n_window_elements, frame_step, delta_time, fdm_method, print_progress),
            obstacles={frame_number: frame.obstacles for frame_number, frame in self.items()}
        )

    def filter_by_person(self, person_id: int) -> "Trajectory":
        filtered_records = OrderedDict(
            (frame_number, frame.persons[person_id])
            for frame_number, frame in self.items()
            if person_id in frame.persons
        )
        return Trajectory(person_id=person_id, records=filtered_records)

    def filter_by_persons(self, person_ids: list[int]) -> "Trajectories":
        filtered_trajectories = {
            person_id: self.filter_by_person(person_id)
            for person_id in person_ids
        }
        return Trajectories(filtered_trajectories)

    def take_first_n(self, n: int) -> "Frames":
        return Frames(OrderedDict(islice(self.items(), n)))

    @classmethod
    def from_trajectories(cls, trajectories: "Trajectories", obstacles: dict[int, list[LineObstacle]]) -> "Frames":
        frame_dict = defaultdict(dict)
        
        for trajectory in trajectories.values():
            for frame_number, person in trajectory.records.items():
                frame_dict[frame_number][trajectory.person_id] = person
        
        frames = {
            frame_number: Frame(number=frame_number, persons=frame_persons, obstacles=obstacles[frame_number])
            for frame_number, frame_persons in frame_dict.items()
        }
        
        return cls(OrderedDict(sorted(frames.items())))

    def to_trajectories(self) -> "Trajectories":
        return Trajectories.from_frames(self)

@dataclass(frozen=True)
class Trajectory:
    person_id: int 
    records: OrderedDict[int, Person]

    def approximate_velocities(
        self, 
        n_window_elements: int, 
        frame_step: int, 
        delta_time: float,
        fdm_method: str = "backward",  # either backward or central 
    ) -> "Trajectory":
        velocities = self._process_frame_windows(
            n_window_elements=n_window_elements,
            frame_step=frame_step,
            window_start=fdm_method,
            valid_func=lambda _: True,
            process_func=lambda lst: Velocity.from_points([p.position for p in lst], delta_time),
        )
        return Trajectory(
            person_id=self.person_id,
            records={
                **{f_num: person.set_velocity(None) for f_num, person in self.records.items()},
                **{f_num: self.records[f_num].set_velocity(vel) for f_num, vel in velocities.items()}
            }
        )

    def approximate_accelerations(
        self, 
        n_window_elements: int, 
        frame_step: int, 
        delta_time: float,
        fdm_method: str = "backward"  # either backward or central 
    ) -> "Trajectory":
        accelerations = self._process_frame_windows(
            n_window_elements=n_window_elements,
            frame_step=frame_step,
            window_start=fdm_method,
            valid_func=lambda p: p.velocity is not None,
            process_func=lambda lst: Acceleration.from_velocities([p.velocity for p in lst], delta_time),
        )
        return Trajectory(
            person_id=self.person_id,
            records={
                **{f_num: person.set_acceleration(None) for f_num, person in self.records.items()},
                **{f_num: self.records[f_num].set_acceleration(acc) for f_num, acc in accelerations.items()}
            }
        )

    def _process_frame_windows(
        self, 
        n_window_elements: int, 
        frame_step: int, 
        window_start: str,
        valid_func: Callable[[Person], bool],
        process_func: Callable[[list[Person]], Any] = lambda _: True
    ) -> dict[int, Any]:
        if window_start == "backward":
            n_window_elements_from_start = n_window_elements - 1  # -1 for the current (middle) element
            n_window_elements_from_end = 1
        elif window_start == "central":
            n_window_elements_from_start = math.ceil(n_window_elements / 2)
            n_window_elements_from_end = math.floor(n_window_elements / 2)
        else:
            raise ValueError("Invalid value for 'window_start'. Use 'backward' or 'central'.")

        frame_numbers = list(self.records.keys())
        records_list = list(self.records.values())
        results = {}

        for i in range(n_window_elements_from_start * frame_step, len(frame_numbers)):
            start_idx = i - n_window_elements_from_start * frame_step
            end_idx = i + n_window_elements_from_end * frame_step
            window_frame_numbers = frame_numbers[start_idx:end_idx:frame_step]

            # Check if the window has the correct spacing
            if len(window_frame_numbers) != n_window_elements or any(
                window_frame_numbers[j] - window_frame_numbers[j - 1] != frame_step
                for j in range(1, len(window_frame_numbers))
            ):
                continue

            window_records = records_list[start_idx:end_idx:frame_step]

            if all(valid_func(p) for p in window_records):
                results[frame_numbers[i]] = process_func(window_records)

        return results

    def get_frames_with_valid_predecessors(self, steps: int, frame_step: int) -> list[int]:
        """
        Returns valid frame numbers that are preceded by a number of valid frames 
        (frames with filled velocity and goal) equal to 'steps'.
        """
        return list(self._process_frame_windows(
            steps, 
            frame_step, 
            window_start="backward",
            valid_func=lambda p: p.velocity is not None and p.goal is not None
        ).keys())

class Trajectories(dict[int, Trajectory]):
    @classmethod
    def from_dict(cls, data_dict):
        obj = cls()
        for k, v in data_dict.items():
            obj[k] = Trajectory(person_id=k, records=v)
        return obj

    def approximate_velocities(
        self, 
        n_window_elements: int, 
        frame_step: int, 
        delta_time: float,
        fdm_method: str = "backward",  # either backward or central 
        print_progress: bool = True
    ) -> "Trajectories":
        return {
            person_id: trajectory.approximate_velocities(n_window_elements, frame_step, delta_time, fdm_method)
            for person_id, trajectory in tqdm(self.items(), desc="Processing frames...", disable=not print_progress)
        }

    def approximate_accelerations(
        self, 
        n_window_elements: int, 
        frame_step: int, 
        delta_time: float,
        fdm_method: str = "backward",  # either backward or central 
        print_progress: bool = True
    ) -> "Trajectories":
        return {
            person_id: trajectory.approximate_accelerations(n_window_elements, frame_step, delta_time, fdm_method)
            for person_id, trajectory in tqdm(self.items(), desc="Processing frames...", disable=not print_progress)
        }

    def filter_by_frame(self, frame_number: int) -> "Frame":
        persons_in_frame = {
            trajectory.person_id: trajectory.records[frame_number]
            for trajectory in self.values()
            if frame_number in trajectory.records
        }
        return Frame(number=frame_number, persons=persons_in_frame)

    def filter_by_frames(self, frame_numbers: list[int]) -> "Frames":
        filtered_frames = {
            frame_number: self.filter_by_frame(frame_number)
            for frame_number in frame_numbers
        }
        return Frames(filtered_frames)

    @classmethod
    def from_frames(cls, frames: Frames) -> "Trajectories":
        trajectories_dict = defaultdict(OrderedDict)
        
        for frame in frames.values():
            for person_id, person in frame.persons.items():
                trajectories_dict[person_id][frame.number] = person

        trajectories = {
            person_id: Trajectory(person_id=person_id, records=OrderedDict(sorted(records.items())))
            for person_id, records in trajectories_dict.items()
        }
        
        return cls(trajectories)

    def to_frames(self, obstacles: dict[int, list[LineObstacle]]) -> Frames:
        return Frames.from_trajectories(self, obstacles)