from dataclasses import dataclass
from functools import cached_property
from entities.vector2d import Point2D, Velocity, Acceleration
from collections import OrderedDict, defaultdict
from typing import Optional, Callable
from tqdm import tqdm
from entities.frame import Frame, Frames
from entities.person import Person
from entities.obstacle import Obstacle
from entities.trajectory import Trajectory, Trajectories
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class Scene:
    id: str
    frames: Frames
    obstacles: list[Obstacle]
    bounding_box: tuple[Point2D, Point2D]
    fps: float
    tag: Optional[list[int]] = None

    # def simulate(
    #     self, 
    #     predict_acc_func: Callable[[list["Features"]], list[Acceleration]],
    #     total_steps: int,
    #     goal_radius: float = 50,
    #     person_ids: Optional[list[int]] = None
    # ) -> "Scene":
    #     from entities.features import Features
    #     frame_numbers = list(self.frames.keys())
    #     frame_step = frame_numbers[1] - frame_numbers[0]
    #     first_frame_persons = self.frames[frame_numbers[0]]
    #     delta_time = frame_step / self.fps
    #     recomputed_frames = {frame_numbers[0]: first_frame_persons}
    #     step = 0
    #     frame_number = frame_numbers[0]
    #     next_frame_number = frame_number + frame_step
    #     simulation_person_out_ids = set()
    #     with tqdm(total=total_steps, initial=step, desc="Simulation processing steps", unit="step") as pbar:
    #         while len(recomputed_frames) > 0 and total_steps > step:
    #             recomputed_frame_no_acc = recomputed_frames[frame_number]
    #             features_dict: dict[int, self.Features] = {
    #                 person_id: Features.get_features(
    #                     person=person,
    #                     person_id=person_id,
    #                     frame=recomputed_frame_no_acc,
    #                     obstacles=self.obstacles
    #                 )
    #                 for person_id, person in recomputed_frame_no_acc.items()
    #                 if (not person_ids or person_id in person_ids)
    #                 and person.velocity is not None and person.goal is not None
    #             }
    #             preds_acc = predict_acc_func(features_dict.values())
    #             recomputed_frame = {**recomputed_frame_no_acc, **{
    #                 person_id: Person(
    #                     position=recomputed_frame_no_acc[person_id].position,
    #                     velocity=recomputed_frame_no_acc[person_id].velocity,
    #                     goal=recomputed_frame_no_acc[person_id].goal,
    #                     acceleration=pred_acc
    #                 )
    #                 for person_id, pred_acc in zip(features_dict.keys(), preds_acc)
    #             }}
    #             recomputed_frames[frame_number] = recomputed_frame
    #             next_recomputed_frame_no_acc = {
    #                 **(
    #                     {
    #                         pid: person 
    #                         for pid, person in self.frames[next_frame_number].items() 
    #                         if pid not in simulation_person_out_ids
    #                     } if next_frame_number in self.frames else {}
    #                     ),
    #                 **{
    #                     person_id: Person(
    #                         position=person.position + person.velocity * delta_time + 0.5 * person.acceleration * delta_time**2,
    #                         velocity=person.velocity + person.acceleration * delta_time,
    #                         goal=person.goal,
    #                         acceleration=Acceleration.zero()
    #                     )
    #                     for person_id, person in recomputed_frame.items()
    #                     if person.velocity is not None and person.goal is not None
    #                 }
    #             }
    #             next_recomputed_frame_no_acc = {
    #                 person_id: person
    #                 for person_id, person in next_recomputed_frame_no_acc.items()
    #                 if person.position.is_within(self.bounding_box[0], self.bounding_box[1]) 
    #                 and not person.position.is_within(person.goal - goal_radius, person.goal + goal_radius)
    #             }
    #             recomputed_frames[next_frame_number] = next_recomputed_frame_no_acc
    #             simulation_person_out_ids.update(recomputed_frame.keys() - next_recomputed_frame_no_acc.keys()) 
    #             step += 1
    #             frame_number = next_frame_number
    #             next_frame_number = frame_number + frame_step
    #             pbar.update(1)
    #     return Scene(
    #         id=self.id,
    #         obstacles=self.obstacles,
    #         frames=recomputed_frames,
    #         fps=self.fps,
    #         tag=self.tag
    #     )

class Scenes(dict[str, Scene]):
    pass