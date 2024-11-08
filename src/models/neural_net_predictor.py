import torch
from models.base_predictor import BasePredictor
from entities.vector2d import Acceleration
from data.scene_collection import SceneCollection
from data.torch_dataset import TorchDataset
from collections import defaultdict
from models.neural_net_model import NeuralNetModel
from entities.scene import Scene

class NeuralNetPredictor(BasePredictor):
    def __init__(self, path: str, batch_size: int = 64, device: str | torch.device = "auto"):
        super().__init__(path)
        self.model.eval()
        self.batch_size = batch_size
        self.device = device

    def predict_scene(self, scene: Scene) -> list[Acceleration]:
        scene_collection = SceneCollection.from_scenes({0: scene})
        return self.predict(scene_collection)[0]

    def predict(self, scene_collection: SceneCollection) -> dict[int, list[Acceleration]]:
        dataset = TorchDataset(scene_collection) 
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=TorchDataset.prepare_batch
        )

        with torch.no_grad():
            raw_preds = self.model.predict(loader, as_numpy=True)

        preds = defaultdict(list)
        for i in range(len(raw_preds)):
            pred = preds[i]
            pred_acc = Acceleration(x=pred[0], y=pred[1])
            
            scene_id, frame_id = dataset.get_scene_and_frame(i)

            preds[scene_id].append(pred_acc)
            assert(len(preds[scene_id]) == frame_id + 1)

        return preds

    def _load_model(self, path: str) -> NeuralNetModel:
        return NeuralNetModel.from_weight_file(path, self.device)