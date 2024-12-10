from dataclasses import dataclass
from entities.person import Person
from entities.obstacle import LineObstacle
from collections import OrderedDict, defaultdict
from typing import Callable
from entities.vector2d import Point2D, Velocity, Acceleration, closest_point_on_line

@dataclass(frozen=True)
class Frame:
    number: int 
    persons: dict[int, Person]
    obstacles: dict[int, LineObstacle]

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
            obstacles={
                obstacle_id: obstacle.normalized(pos_scale, vel_scale, acc_scale)
                for obstacle_id, obstacle in self.obstacles.items()
            }
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

    def filter_invalid_persons(self) -> "Frame":
        return Frame(
            number=self.number,
            persons={pid: person for pid, person in self.persons.items() if person.goal and person.velocity},
            obstacles=self.obstacles
        )

    def apply_kinematic_equation(self, delta_time: float) -> "Frame":
        return Frame(
            number=self.number,
            persons={pid: person.apply_kinematic_equation(delta_time) for pid, person in self.persons.items()},
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
        for line in self.obstacles.values():
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
        return OrderedDict({
            frame_number: frame.normalized(pos_scale, vel_scale, acc_scale)
            for frame_number, frame in self.items()
        })

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

    @classmethod
    def from_trajectories(cls, trajectories: "Trajectories", obstacles: dict[int, LineObstacle]) -> "Frames":
        frame_dict = defaultdict(dict)
        
        for trajectory in trajectories.values():
            for frame_number, person in trajectory.records.items():
                frame_dict[frame_number][trajectory.person_id] = person
        
        frames = {
            frame_number: Frame(number=frame_number, persons=frame_persons, obstacles=obstacles)
            for frame_number, frame_persons in frame_dict.items()
        }
        
        return cls(OrderedDict(sorted(frames.items())))

    def to_trajectories(self) -> "Trajectories":
        return Trajectories.from_frames(self)

@dataclass(frozen=True)
class Trajectory:
    person_id: int 
    records: OrderedDict[int, Person]

    def get_pred_valid_frame_numbers(self, steps: int, frame_step: int) -> list[int]:
        if len(self.records) == 0:
            return []

        frame_numbers = list(self.records.keys())
        segments = []

        for i in range(len(frame_numbers)):
            frame_number = frame_numbers[i]
            person_data = self.records[frame_number]
            person_valid = person_data.velocity is not None and person_data.goal is not None
            
            if person_valid:
                last_segment_end = segments[-1][1] if segments else None

                if last_segment_end is not None and frame_number == last_segment_end + frame_step:
                    # Extend the current segment
                    segments[-1] = (segments[-1][0], frame_number)
                else:
                    # Start a new segment
                    segments.append((frame_number, frame_number))

        valid_frame_numbers = []
        for start, end in segments:
            # Compute valid starting frames in the segment
            valid_starts = range(start, end - (steps - 1) * frame_step, frame_step)
            valid_frame_numbers.extend(valid_starts)

        return valid_frame_numbers

class Trajectories(dict[int, Trajectory]):
    @classmethod
    def from_dict(cls, data_dict):
        obj = cls()
        for k, v in data_dict.items():
            obj[k] = Trajectory(person_id=k, records=v)
        return obj

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

    def to_frames(self, obstacles: dict[int, LineObstacle]) -> Frames:
        return Frames.from_trajectories(self, obstacles)