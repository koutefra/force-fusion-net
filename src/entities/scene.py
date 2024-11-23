from dataclasses import dataclass
from functools import cached_property
from entities.vector2d import Point2D, Velocity, Acceleration
from entities.obstacle import BaseObstacle, PointObstacle, LineObstacle
from collections import OrderedDict, defaultdict
from typing import Optional, Callable
from tqdm import tqdm

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

    @cached_property
    def bounding_box(self) -> tuple[Point2D, Point2D]:
        x_min, y_min = float("inf"), float("inf")
        x_max, y_max = float("-inf"), float("-inf")
        
        # persons
        for persons in self.frames.values():
            for person in persons.values():
                pos = person.position
                x_min, y_min = min(x_min, pos.x), min(y_min, pos.y)
                x_max, y_max = max(x_max, pos.x), max(y_max, pos.y)
                
        # obstacles
        for obstacle in self.obstacles:
            if isinstance(obstacle, PointObstacle):
                pos = obstacle.position
                x_min, y_min = min(x_min, pos.x), min(y_min, pos.y)
                x_max, y_max = max(x_max, pos.x), max(y_max, pos.y)
            elif isinstance(obstacle, LineObstacle):
                for vertex in obstacle.line:  # Assuming `line` is a tuple of Point2D
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
        predict_acc_func: Callable[[list["Features"]], list[Acceleration]],
        total_steps: int,
        goal_radius: float = 100,
        person_ids: Optional[list[int]] = None
    ) -> "Scene":
        from entities.features import Features
        frame_numbers = list(self.frames.keys())
        frame_step = frame_numbers[1] - frame_numbers[0]
        first_frame_persons = self.frames[frame_numbers[0]]
        delta_time = frame_step / self.fps
        recomputed_frames = {frame_numbers[0]: first_frame_persons}

        step = 0
        frame_number = frame_numbers[0]
        next_frame_number = frame_number + frame_step
        with tqdm(total=total_steps, initial=step, desc="Simulation processing steps", unit="step") as pbar:
            while len(recomputed_frames) > 0 and total_steps > step:
                recomputed_frame_no_acc = recomputed_frames[frame_number]

                features_dict: dict[int, self.Features] = {
                    person_id: Features.get_features(
                        person=person,
                        person_id=person_id,
                        frame=recomputed_frame_no_acc,
                        obstacles=self.obstacles
                    )
                    for person_id, person in recomputed_frame_no_acc.items()
                    if (not person_ids or person_id in person_ids)
                    and person.velocity is not None and person.goal is not None
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
                recomputed_frames[frame_number] = recomputed_frame

                next_recomputed_frame_no_acc = {
                    **(self.frames[next_frame_number] if next_frame_number in self.frames else {}),  # add newcomers
                    **{
                        person_id: Person(
                            position=person.position + person.velocity * delta_time + 0.5 * person.acceleration * delta_time**2,
                            velocity=person.velocity + person.acceleration * delta_time,
                            goal=person.goal,
                            acceleration=Acceleration.zero()
                        )
                        for person_id, person in recomputed_frame.items()
                        if person.velocity is not None and person.goal is not None
                    }
                }

                # import numpy as np
                # print(
                #     np.mean(
                #         [
                #             person.acceleration.magnitude() 
                #             for pid, person in recomputed_frame.items()
                #             if person.acceleration
                #         ]
                #     )
                # )

                # filter persons out of the scene
                next_recomputed_frame_no_acc = {
                    person_id: person
                    for person_id, person in next_recomputed_frame_no_acc.items()
                    if person.position.is_within(self.bounding_box[0], self.bounding_box[1]) 
                    and not person.position.is_within(person.goal - goal_radius, person.goal + goal_radius)
                }
                recomputed_frames[next_frame_number] = next_recomputed_frame_no_acc

                step += 1
                frame_number = next_frame_number
                next_frame_number = frame_number + frame_step
                pbar.update(1)
                
        return Scene(
            id=self.id,
            obstacles=self.obstacles,
            frames=recomputed_frames,
            fps=self.fps,
            tag=self.tag
        )

Scenes = dict[str, Scene]
