from dataclasses import dataclass
from entities.vector2d import Point2D, Velocity, Acceleration
from entities.obstacle import BaseObstacle
from collections import OrderedDict, defaultdict
from typing import Optional

@dataclass(frozen=True)
class Person:
    position: Point2D
    goal: Optional[Point2D] = None
    velocity: Optional[Velocity] = None
    acceleration: Optional[Acceleration] = None

Frame = dict[int, Person]  # person_id -> Person
Frames = OrderedDict[int, Frame]  # frame_number -> Frame
Trajectory = OrderedDict[int, Person]  # frane_number -> Person
Trajectories = dict[int, Trajectory]  # person_id -> Trajectory

@dataclass(frozen=True)
class Scene:
    id: str
    obstacles: list[BaseObstacle]
    frames: Frames
    fps: float
    tag: Optional[list[int]] = None

    @property
    def bounding_box(self) -> tuple[Point2D, Point2D]:
        x_min, y_min = float("inf"), float("inf")
        x_max, y_max = float("-inf"), float("-inf")
        
        # persons
        for frame in self.frames.values():
            for person_in_frame in frame.persons.values():
                pos = person_in_frame.position
                x_min, y_min = min(x_min, pos.x), min(y_min, pos.y)
                x_max, y_max = max(x_max, pos.x), max(y_max, pos.y)
                
        # obstacles
        for obstacle in self.obstacles:
            for vertex in obstacle.vertices:
                x_min, y_min = min(x_min, vertex.x), min(y_min, vertex.y)
                x_max, y_max = max(x_max, vertex.x), max(y_max, vertex.y)
        
        return Point2D(x=x_min, y=y_min), Point2D(x=x_max, y=y_max)

    @staticmethod
    def trajectories_to_frames(trajectories: Trajectories) -> Frames:
        frame_to_tracks = defaultdict(dict)
        for person_id, person_frames in trajectories.items():
            for frame_number, person_data in person_frames.items():
                frame_to_tracks[frame_number][person_id] = person_data
        ordered_frame_to_tracks = OrderedDict(sorted(frame_to_tracks.items()))
        return ordered_frame_to_tracks

    # def simulate(
    #     self, 
    #     person_ids: list[int],
    #     predict_force_func: Callable[[dict[int, FrameObject], PersonInFrame, Point2D], Acceleration]
    # ) -> "Scene":
    #     frame_numbers = list(self.frames.keys())
    #     recomputed_frames = {frame_numbers[0]: self.frames[frame_numbers[0]]}
    #     for frame_id, cur_frame_number in enumerate(frame_numbers):
    #         cur_frame_objects = {}
    #         next_frame_objects_wo_acc = {}
    #         cur_frame_objects_wo_acc = self.frames[cur_frame_number].frame_objects
    #         common_person_ids = set(person_ids).intersection(cur_frame_objects_wo_acc)

    #         # compute accelerations
    #         for person_id in common_person_ids:
    #             person_in_frame_no_acc = cur_frame_objects_wo_acc[person_id]
    #             pred_acc = predict_force_func(
    #                 cur_frame_objects_wo_acc, 
    #                 person_in_frame_no_acc, 
    #                 self.persons[person_id].goal
    #             )
    #             person_in_frame = PersonInFrame(
    #                 id=person_id,
    #                 position=person_in_frame_no_acc.position,
    #                 velocity=person_in_frame_no_acc.velocity,
    #                 acceleration=pred_acc
    #             )
    #             cur_frame_objects[person_id] = person_in_frame

    #             person_in_next_frame_no_acc = PersonInFrame(
    #                 id=person_id,
    #                 position=person_in_frame.position + person_in_frame.velocity * delta_time,
    #                 velocity=person_in_frame.velocity + person_in_frame.acceleration * delta_time,
    #                 acceleration=Acceleration.zero()
    #             )
    #             next_frame_objects_wo_acc[person_id] = person_in_next_frame_no_acc

    #         recomputed_frames[cur_frame_number] = Frame(
    #             number=cur_frame_number, 
    #             frame_objects=cur_frame_objects
    #         )

    #         if frame_id < len(self.frames) - 1:
    #             # go forward in time
    #             next_frame_number = frame_numbers[frame_id + 1]
    #             delta_time = (next_frame_number - cur_frame_number) / self.fps
    #             recomputed_frames[next_frame_number] = Frame(
    #                 number=next_frame_number, 
    #                 frame_objects=next_frame_objects_wo_acc
    #             )

    #     return Scene(
    #         id=self.id,
    #         obstacles=self.obstacles,
    #         persons=self.persons,
    #         frames=recomputed_frames,
    #         fps=self.fps,
    #         tag=self.tag,
    #         dataset=self.dataset
    #     )

Scenes = dict[str, Scene]