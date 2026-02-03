"""Helper functions for GSO benchmark."""

import os


def get_gso_instance_docker_image(instance_id: str) -> str:
    """Get the Docker image for a GSO instance."""
    prefix = (os.environ.get("GSO_IMAGE_PREFIX") or "docker.io/slimshetty/gso").rstrip("/")
    return f"{prefix}:gso.eval.x86_64.{instance_id}".lower()


def get_gso_workspace_dir_name(instance: dict) -> str:
    """Get the workspace directory name for an instance."""
    return instance["repo"].replace("/", "__")
