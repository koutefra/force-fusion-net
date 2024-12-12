from dataclasses import dataclass
from entities.vector2d import Point2D, Velocity, Acceleration
from typing import Optional, Callable
from tqdm import tqdm
from entities.frame import Frames, Frame

@dataclass(frozen=True)
class Scene:
    id: str
    frames: Frames
    bounding_box: tuple[Point2D, Point2D]
    fps: float
    frame_step: int = 1
    tag: Optional[list[int]] = None

    def normalized(
        self, 
        pos_scale: Callable[[Point2D], Point2D] = lambda x: x, 
        vel_scale: Callable[[Velocity], Velocity] = lambda v: v, 
        acc_scale: Callable[[Acceleration], Acceleration] = lambda a: a) -> "Scene":
        return Scene(
            id=self.id,
            frames=self.frames.normalized(pos_scale, vel_scale, acc_scale),
            bounding_box=(pos_scale(self.bounding_box[0], pos_scale(self.bounding_box[1]))),
            fps=self.fps,
            frame_step=self.frame_step,
            tag=self.tag
        )

    def simulate(
        self,
        predict_acc_func: Callable[[Frame], list[Acceleration]],
        total_steps: int,
        goal_radius: float,
        person_ids: Optional[list[int]] = None
    ) -> "Scene":
        frame_numbers = list(self.frames.keys())
        first_frame_number, first_frame = frame_numbers[0], self.frames[frame_numbers[0]]
        resulting_frames = [first_frame.filter_invalid_persons().remove_persons(person_ids if person_ids else [])]
        delta_time = self.frame_step / self.fps
        finished_person_ids = set(person_ids or [])
        for step in tqdm(
            range(first_frame_number, first_frame_number + total_steps * self.frame_step, self.frame_step), 
            desc="Computing the simulation..."
        ):
            frame = resulting_frames[-1]
            acc_pred = predict_acc_func(frame)
            frame = frame.set_accelerations(acc_pred)
            resulting_frames[-1] = frame  # resave the currect frame enriched with the accelerations

            next_frame = frame.apply_kinematic_equation(delta_time)
            finished_person_ids.update(next_frame.get_person_ids_at_goal(goal_radius))
            finished_person_ids.update(next_frame.get_person_ids_outside(self.bounding_box))

            new_persons = {}
            if step + self.frame_step in self.frames:
                new_persons = self.frames[step + self.frame_step].filter_invalid_persons().persons
            new_persons = {pid: person for pid, person in new_persons.items() if pid not in finished_person_ids}

            next_frame = next_frame.add_persons(new_persons).remove_persons(finished_person_ids)
            resulting_frames.append(next_frame)

        return Scene(
            id=self.id,
            frames={first_frame_number + i * self.frame_step: frame for frame, i in zip(resulting_frames, range(len(resulting_frames)))},
            bounding_box=self.bounding_box,
            fps=self.fps,
            frame_step=self.frame_step,
            tag=self.tag
        )

    def take_first_n_frames(self, n: int) -> "Scene":
        return Scene(
            id=self.id,
            frames=self.frames.take_first_n(n),
            bounding_box=self.bounding_box,
            fps=self.fps,
            frame_step=self.frame_step,
            tag=self.tag
        )

class Scenes(dict[str, Scene]):
    def normalized(
        self, 
        pos_scale: Callable[[Point2D], Point2D] = lambda x: x, 
        vel_scale: Callable[[Velocity], Velocity] = lambda v: v, 
        acc_scale: Callable[[Acceleration], Acceleration] = lambda a: a) -> "Scenes":
        return {
            scene_id: scene.normalized(pos_scale, vel_scale, acc_scale)
            for scene_id, scene in self.items()
        }