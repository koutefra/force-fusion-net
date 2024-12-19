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

    def approximate_velocities(self, n_window_elements: int, fdm_method: str, print_progress: bool = True) -> "Scene":
        if print_progress:
            print(f'Scene {self.id}: calculating velocities...')
        delta_time = 1 / self.fps
        return Scene(
            id=self.id,
            frames=self.frames.approximate_velocities(n_window_elements, self.frame_step, delta_time, fdm_method, print_progress),
            bounding_box=self.bounding_box,
            fps=self.fps,
            frame_step=self.frame_step,
            tag=self.tag
        )

    def approximate_accelerations(self, n_window_elements: int, fdm_method: str, print_progress: bool = True) -> "Scene":
        if print_progress:
            print(f'Scene {self.id}: calculating accelerations...')
        delta_time = 1 / self.fps
        return Scene(
            id=self.id,
            frames=self.frames.approximate_accelerations(n_window_elements, self.frame_step, delta_time, fdm_method, print_progress),
            bounding_box=self.bounding_box,
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
        first_f_num, first_frame = frame_numbers[0], self.frames[frame_numbers[0]]
        last_f_num = first_f_num + total_steps * self.frame_step
        resulting_frames = [first_frame.remove_persons(person_ids) if person_ids else first_frame]
        delta_time = self.frame_step / self.fps
        finished_person_ids = set(person_ids or [])
        for cur_f_num in tqdm(range(first_f_num, last_f_num + 1, self.frame_step), desc="Calculating the simulation..."):
            frame = resulting_frames[-1]
            acc_pred = predict_acc_func(frame)
            frame = frame.set_accelerations(acc_pred)
            resulting_frames[-1] = frame  # resave the currect frame enriched with the accelerations

            next_frame = frame.apply_kinematic_equation(delta_time, self.frame_step)
            finished_person_ids.update(next_frame.get_person_ids_at_goal(goal_radius))
            finished_person_ids.update(next_frame.get_person_ids_outside(self.bounding_box))

            # add new persons from the original scene if available
            if cur_f_num + self.frame_step in self.frames:
                next_frame = next_frame.add_persons(self.frames[cur_f_num + self.frame_step].persons)
            next_frame = next_frame.remove_persons(finished_person_ids)

            # add next frame for further
            if cur_f_num != last_f_num:
                resulting_frames.append(next_frame)

        return Scene(
            id=self.id,
            frames=Frames({
                f_num: frame 
                for frame, f_num in zip(resulting_frames, range(first_f_num, last_f_num, self.frame_step))
            }),
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

    def approximate_velocities(self, n_window_elements: int, fdm_method: str, print_progress: bool = True) -> "Scenes":
        return {
            scene_id: scene.approximate_velocities(n_window_elements, fdm_method, print_progress=False)
            for scene_id, scene in tqdm(self.items(), f"Calculating velocities using FDM {fdm_method}", disable=not print_progress)
        }

    def approximate_accelerations(self, n_window_elements: int, fdm_method: str, print_progress: bool = True) -> "Scene":
        return {
            scene_id: scene.approximate_accelerations(n_window_elements, fdm_method, print_progress=False)
            for scene_id, scene in tqdm(self.items(), f"Calculating accelerations using FDM {fdm_method}", disable=not print_progress)
        }