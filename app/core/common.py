from helpers.checkpoint import SenkuCheckpointManager


def list_available_checkpoints():
    return SenkuCheckpointManager().list_checkpoints()
