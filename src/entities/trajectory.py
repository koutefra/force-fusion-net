from dataclasses import dataclass
from entities.person import Person
from entities.frame import Frame, Frames
from entities.vector2d import Point2D, Velocity
from collections import OrderedDict, defaultdict

@dataclass(frozen=True)
class Trajectory:
    person_id: int 
    records: OrderedDict[int, Person]

@dataclass(frozen=True)
class Trajectories(dict[int, Trajectory]):
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
