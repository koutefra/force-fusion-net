import torch
from models.base_predictor import BasePredictor
from entities.vector2d import Acceleration
from data.scene_collection import SceneCollection
from data.torch_dataset import TorchDataset
from collections import defaultdict
from models.neural_net_model import NeuralNetModel

class NeuralNetPredictor(BasePredictor):
    def __init__(self, model: NeuralNetModel, batch_size: int):
        super().__init__(model)
        self.model.eval()
        self.batch_size = batch_size

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