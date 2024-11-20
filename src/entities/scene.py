from dataclasses import dataclass
from functools import cached_property
from entities.vector2d import Point2D, Velocity, Acceleration
from entities.obstacle import BaseObstacle
from collections import OrderedDict, defaultdict
from typing import Optional, Callable

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
    from entities.features import Features

    id: str
    obstacles: list[BaseObstacle]
    frames: Frames
    fps: float
    tag: Optional[list[int]] = None

    @cached_property
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

    def simulate(
        self, 
        predict_acc_func: Callable[[list[Features]], list[Acceleration]],
        frame_step: int,
        total_steps: int,
        person_ids: Optional[list[int]]
    ) -> "Scene":
        first_frame_number = list(self.frames.keys())[0]
        first_frame_persons = self.frames[first_frame_number]
        delta_time = frame_step / self.fps
        recomputed_frames = [first_frame_persons]

        step = 1
        while len(recomputed_frames) > 0 and total_steps >= step:
            recomputed_frame_no_acc = recomputed_frames[-1]

            features_dict: dict[int, self.Features] = {
                person_id: self.Features.get_features(
                    person=person,
                    person_id=person_id,
                    frame=recomputed_frame,
                    obstacles=self.obstacles
                )
                for person_id, person in recomputed_frame_no_acc.items()
                if not person_ids or person_id in person_ids
            }

            preds_acc = predict_acc_func(features_dict.values())

            recomputed_frame = {**recomputed_frame_no_acc, **{
                person_id: Person(
                    position=recomputed_frame_no_acc[person_id].position,
                    velocity=recomputed_frame_no_acc[person_id].velocity,
                    goal=recomputed_frame_no_acc[person_id].goal,
                    acceleration=pred_acc
                )
                for person_id, pred_acc in zip(features_dict.keys(), preds_acc)
            }}
            recomputed_frames[-1] = recomputed_frame

            next_recomputed_frame_no_acc = {
                **(self.frames[step] if step in self.frames else {}),  # add newcomers
                **{
                    person_id: Person(
                        position=person.position + person.velocity * delta_time,
                        velocity=person.velocity + person.acceleration * delta_time,
                        goal=person.goal,
                        acceleration=Acceleration.zero()
                    )
                    for person_id, person in recomputed_frame.items()
                }
            }

            # filter persons out of the scene
            next_recomputed_frame_no_acc = {
                person_id: person
                for person_id, person in next_recomputed_frame_no_acc.items()
                if person.position.is_within(self.bounding_box[0], self.bounding_box[1]) 
            }
            recomputed_frames.append(next_recomputed_frame_no_acc)

        recomputed_frames_dict = {
            first_frame_number + i * frame_step: frame for i, frame in enumerate(recomputed_frames)
        }
                
        return Scene(
            id=self.id,
            obstacles=self.obstacles,
            frames=recomputed_frames_dict,
            fps=self.fps,
            tag=self.tag
        )

Scenes = dict[str, Scene]