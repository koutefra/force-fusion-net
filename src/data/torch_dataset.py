import torch
from entities.scene import Scenes
from entities.frame import Frames
from entities.batched_frames import BatchedFrames

class TorchSceneDataset(torch.utils.data.Dataset):
    def __init__(self, scenes: Scenes, steps: int, device: torch.device, dtype: torch.dtype):
        self.scenes = scenes
        self.steps = steps
        self.device = device
        self.dtype = dtype
        self._mapping = self._compute_id_scene_mapping(scenes, steps)

    @staticmethod
    def _compute_id_scene_mapping(scenes: Scenes, steps: int) -> list[int]:
        mapping = []
        for scene_id, scene in scenes.items():
            f_step = scene.frame_step
            for person_id, trajectory in scene.frames.to_trajectories().items():
                for frame_number in trajectory.get_pred_valid_frame_numbers(steps, f_step):
                    mapping.append((scene_id, person_id, frame_number))
        return mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def __getitem__(self, idx: int) -> tuple[tuple[Frames, int], tuple[torch.Tensor, int]]:
        scene_id, person_id, start_f_number = self._mapping[idx]
        scene = self.scenes[scene_id]
        f_step = scene.frame_step
        last_f_number = start_f_number + self.steps * f_step
        frames = {
            frame_number: scene.frames[frame_number]
            for frame_number in range(start_f_number, last_f_number + 1, f_step)
        }
        last_position = frames[last_f_number].persons[person_id].position
        delta_time = torch.tensor(1 / scene.fps, device=self.device, dtype=self.dtype)
        return (frames, person_id), (last_position.to_tensor(self.device, self.dtype), delta_time)

    def prepare_batch(
        self, 
        data: list[tuple[tuple[Frames, int], tuple[torch.Tensor, int]]]
    ) -> tuple[tuple[BatchedFrames, torch.Tensor], torch.Tensor]:
        inputs, labels = zip(*data)
        ground_truths, delta_times = zip(*labels)
        ground_truths_tensor = torch.stack(ground_truths).to(self.device, dtype=self.dtype)
        delta_times_tensor = torch.stack(delta_times).to(self.device, dtype=self.dtype)

        frames_list, person_ids_list = zip(*inputs)
        batched_frames = BatchedFrames(frames_list, person_ids_list, self.device, self.dtype)

        return (batched_frames, delta_times_tensor), ground_truths_tensor