from app.helpers.checkpoint import SenkuCheckpointManager, SenkuCheckpoint


def get_checkpoint(checkpoint_name: str) -> SenkuCheckpoint:
    return SenkuCheckpointManager().get_checkpoint(checkpoint_name)


def list_available_checkpoints():
    return SenkuCheckpointManager().list_checkpoints()
