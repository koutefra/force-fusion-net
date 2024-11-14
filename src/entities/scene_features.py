class DatapointFeatures:
    individual_features: dict[str, float]
    interaction_features: list[dict[str, float]]

SceneFeatures = list[DatapointFeatures]