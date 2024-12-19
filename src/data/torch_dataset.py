import torch
from entities.scene import Scenes
from entities.frame import Frame
from entities.batched_frames import BatchedFrames

class TorchSceneDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        scenes: Scenes, 
        pred_steps: int, 
        device: torch.device, 
        dtype: torch.dtype
    ):
        self.scenes = scenes
        self.pred_steps = pred_steps
        self.device = device
        self.dtype = dtype
        self._mapping = self._compute_id_scene_mapping(scenes, pred_steps)

    @staticmethod
    def _compute_id_scene_mapping(scenes: Scenes, steps: int) -> list[int]:
        mapping = []
        for scene_id, scene in scenes.items():
            f_step = scene.frame_step
            for person_id, trajectory in scene.frames.to_trajectories().items():
                for frame_number in trajectory.get_frames_with_valid_predecessors(steps + 1, f_step):
                    mapping.append((scene_id, person_id, frame_number))
        return mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def __getitem__(self, idx: int) -> tuple[list[Frame], int, float]:
        scene_id, person_id, last_f_number = self._mapping[idx]
        scene = self.scenes[scene_id]
        f_step = scene.frame_step
        start_f_number = last_f_number - self.pred_steps * f_step
        delta_time = 1 / scene.fps
        frames = [
            scene.frames[frame_number]
            for frame_number in range(start_f_number, last_f_number + 1, f_step)
        ]
        return frames, person_id, delta_time

    def prepare_batch(self, data: list[tuple[list[Frame], int, int]]) -> tuple[BatchedFrames, torch.Tensor]:
        frames, person_ids, delta_times = zip(*data)
        batched_frames = BatchedFrames(frames, person_ids, delta_times, self.device, dtype=self.dtype)
        ground_truths = batched_frames.get_gt_next_positions()
        if ground_truths is not None and not ground_truths.is_contiguous():
            ground_truths = ground_truths.contiguous()
        return batched_frames, ground_truths