from entities.scene import Scenes
from data.loaders.base_loader import BaseLoader
from typing import Callable
from entities.vector2d import Point2D, Velocity, Acceleration

class SceneDataset:
    scenes = Scenes()

    def __init__(self, loaders: list[BaseLoader], print_progress: bool = True):
        self.loaders = loaders
        self.scenes = self._load(loaders, print_progress)
        self.print_progress = print_progress

    @staticmethod
    def _load(loaders: list[BaseLoader], print_progress: bool = True):
        scenes = Scenes()
        for loader in loaders:
            loaded_scenes = loader.load(print_progress)
            scenes = Scenes({**scenes, **loaded_scenes})
        return scenes

    def normalized(
        self, 
        pos_scale: Callable[[Point2D], Point2D] = lambda x: x, 
        vel_scale: Callable[[Velocity], Velocity] = lambda v: v, 
        acc_scale: Callable[[Acceleration], Acceleration] = lambda a: a) -> "SceneDataset":
        scene_dataset = SceneDataset(loaders=[], print_progress=self.print_progress)
        scene_dataset.scenes = self.scenes.normalized(pos_scale, vel_scale, acc_scale)
        return scene_dataset

    def split(self, val_ids: list[str]) -> tuple["SceneDataset", "SceneDataset"]:
        train_ids = set(self.scenes.keys()) - set(val_ids)
        train_scenes = Scenes({scene_id: self.scenes[scene_id] for scene_id in train_ids if scene_id in self.scenes})
        val_scenes = Scenes({scene_id: self.scenes[scene_id] for scene_id in val_ids if scene_id in self.scenes})

        train_dataset = SceneDataset(loaders=[], print_progress=self.print_progress)
        train_dataset.scenes = train_scenes

        val_dataset = SceneDataset(loaders=[], print_progress=self.print_progress)
        val_dataset.scenes = val_scenes

        return train_dataset, val_dataset