from pathlib import Path

import mlflow


def create_mlflow_experiment(experiment_name: str, mode: oct = 0o777) -> str:
    """
    Set mlflow experiment and permissions for folder.

    Args:
        experiment_name: experiment name
        mode: experiment name mode, default = 0o777
    """
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    artifact_location = Path(experiment.artifact_location)

    if not artifact_location.exists():
        artifact_location.mkdir()
        artifact_location.chmod(mode)

    return experiment.experiment_id
