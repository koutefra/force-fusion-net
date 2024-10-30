from typing import List
from core.scene import Scene
from data.trajnetplusplus_loader import TrajnetPlusPlusLoader
import pickle
import os
import torch

class PedestrianDataset:
    scenes: List[Scene]

    def __init__(self, scenes: List[Scene]):
        self.scenes = scenes

    @classmethod
    def from_dataset(cls, path: str, dataset_type: str, dataset_name: str, cache_path: str = None) -> "PedestrianDataset":
        """Load a dataset based on the type provided, with optional caching."""
        if cache_path and os.path.exists(cache_path):
            print("Loading dataset from cache...")
            return cls.load_from_cache(cache_path)

        # Proceed with regular loading
        if dataset_type == "trajnet++":
            loader = TrajnetPlusPlusLoader(dataset_name)
            loader.load(path)
            scenes = loader.preprocess()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_name}")

        dataset = cls(scenes=scenes)
        
        # Save to cache if cache_path is provided
        if cache_path:
            dataset.save_to_cache(cache_path)
        
        return dataset

    def save_to_cache(self, cache_path: str):
        """Save the preprocessed dataset to a cache file."""
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(self, cache_file)
        print(f"Dataset saved to cache at {cache_path}.")

    def to_torch_dataset(self, model_name: str = None) -> torch.utils.data.Dataset:
        """Convert to a PyTorch-compatible Dataset, optionally loading predictions for a model."""
        # return TorchPedestrianDataset(self, model_name=model_name)
        raise NotImplementedError()

    @classmethod
    def load_from_cache(cls, cache_path: str) -> "PedestrianDataset":
        """Load the preprocessed dataset from a cache file."""
        with open(cache_path, 'rb') as cache_file:
            dataset = pickle.load(cache_file)
        print(f"Dataset loaded from cache at {cache_path}.")
        return dataset

def main(path: str, dataset_type: str, dataset_name: str, cache_path: str = None):
    print(f"Loading dataset '{dataset_name}' from path: {path}")
    dataset = PedestrianDataset.from_dataset(path, dataset_type, dataset_name, cache_path)
    
    # Display some information about the loaded scenes
    print(f"Number of scenes loaded: {len(dataset.scenes)}")
    for i, scene in enumerate(dataset.scenes[:3]):  # Show the first 3 scenes as a sample
        print(f"Scene {i+1}:")
        print(f"  ID: {scene.id}")
        print(f"  Focus Person IDs: {scene.focus_person_ids}")
        print(f"  FPS: {scene.fps}")
        print(f"  Tag: {scene.tag}")
        print(f"  Number of Trajectories: {len(scene.trajectories)}")
        print(f"  Dataset: {scene.dataset}")
        print("")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pedestrian Dataset Loader and Showcase")
    parser.add_argument("--path", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--dataset_type", type=str, default="trajnet++", help="Type of the dataset type")
    parser.add_argument("--dataset_name", type=str, default='unknown', help="Name of the dataset type")
    parser.add_argument("--cache_path", type=str, default=None, help="Path to save or load the cache file")

    args = parser.parse_args()
    main(args.path, args.dataset_type, args.dataset_name, args.cache_path)

# import torch
# class TorchPedestrianDataset(Dataset):
#     def __init__(self, pedestrian_dataset: PedestrianDataset, model_name: str = None):
#         self.pedestrian_dataset = pedestrian_dataset
#         self.model_name = model_name
    
#     def __len__(self) -> int:
#         return len(self.pedestrian_dataset.scenes)
    
#     def __getitem__(self, index: int) -> dict:
#         scene = self.pedestrian_dataset.scenes[index]
        
#         # Convert positions and optional predictions to tensors
#         positions = torch.tensor([[pos.x, pos.y] for pos in scene.positions.values()])
#         velocities = (
#             scene.get_predictions(self.model_name, "velocities") if self.model_name else None
#         )
#         velocities_tensor = (
#             torch.tensor([[vec.x, vec.y] for vec in velocities.values()])
#             if velocities else torch.zeros_like(positions)
#         )

#         return {
#             "positions": positions,
#             "velocities": velocities_tensor,
#             "scene_id": scene.id
#         }