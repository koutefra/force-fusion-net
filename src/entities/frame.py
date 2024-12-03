from dataclasses import dataclass
from entities.person import Person
from entities.trajectory import Trajectory, Trajectories
from entities.obstacle import Obstacle
from collections import OrderedDict, defaultdict
from typing import Optional

@dataclass(frozen=True)
class Frame:
    number: int 
    persons: dict[int, Person]

    def get_all_features(
        self, 
        person_id: int, 
        obstacles: list[Obstacle]
    ) -> tuple[list[float], list[list[float]], list[list[float]]]:
        person = self.persons[person_id]
        individual_features = person.get_individual_features()
        interaction_features = self.get_interaction_features(person_id, person)
        obstacle_features = person.get_obstacle_features(obstacles)
        return individual_features, interaction_features, obstacle_features

    def get_interaction_features(self, person_id: int, person: Optional[Person] = None) -> list[list[float]]:
        person = person if person else self.persons[person_id]
        interaction_features = []
        for other_person_id, other_person in self.persons.items():
            if person_id == other_person_id or other_person.velocity is None:
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

class Frames(OrderedDict[int, Frame]):
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
    def from_trajectories(cls, trajectories: Trajectories) -> "Frames":
        frame_dict = defaultdict(dict)
        
        for trajectory in trajectories.values():
            for frame_number, person in trajectory.records.items():
                frame_dict[frame_number][trajectory.person_id] = person
        
        frames = {
            frame_number: Frame(number=frame_number, persons=frame_persons)
            for frame_number, frame_persons in frame_dict.items()
        }
        
        return cls(OrderedDict(sorted(frames.items())))