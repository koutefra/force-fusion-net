from dataclasses import dataclass
from entities.scene import Scenes
from data.base_loader import BaseLoader
from typing import Callable
from entities.vector2d import Point2D, Velocity, Acceleration

@dataclass(frozen=True)
class SceneDataset:
    scenes: Scenes

    @classmethod
    def from_loaders(cls, loaders: list[BaseLoader], print_progress: bool = True) -> "SceneDataset":
        """Creates a SceneDataset by loading scenes from multiple loaders."""
        scenes = cls._load(loaders, print_progress)
        return cls(scenes=scenes)

    @staticmethod
    def _load(loaders: list[BaseLoader], print_progress: bool = True) -> Scenes:
        """Loads scenes from a list of BaseLoader instances."""
        scenes = Scenes()
        for loader in loaders:
            loaded_scenes = loader.load(print_progress)
            scenes = Scenes({**scenes, **loaded_scenes})
        return scenes

    def approximate_velocities(self, n_window_elements: int, fdm_method: str, print_progress: bool = True) -> "SceneDataset":
        return SceneDataset(self.scenes.approximate_velocities(n_window_elements, fdm_method, print_progress))

    def approximate_accelerations(self, n_window_elements: int, fdm_method: str, print_progress: bool = True) -> "SceneDataset":
        return SceneDataset(self.scenes.approximate_accelerations(n_window_elements, fdm_method, print_progress))

    def normalized(
        self, 
        pos_scale: Callable[[Point2D], Point2D] = lambda x: x, 
        vel_scale: Callable[[Velocity], Velocity] = lambda v: v, 
        acc_scale: Callable[[Acceleration], Acceleration] = lambda a: a) -> "SceneDataset":
        return SceneDataset(
            self.scenes.normalized(pos_scale, vel_scale, acc_scale)
        )

    def split(self, val_ids: list[str]) -> tuple["SceneDataset", "SceneDataset"]:
        train_ids = set(self.scenes.keys()) - set(val_ids)
        train_scenes = Scenes({scene_id: self.scenes[scene_id] for scene_id in train_ids if scene_id in self.scenes})
        val_scenes = Scenes({scene_id: self.scenes[scene_id] for scene_id in val_ids if scene_id in self.scenes})
        train_dataset = SceneDataset(train_scenes)
        val_dataset = SceneDataset(val_scenes)
        return train_dataset, val_dataset