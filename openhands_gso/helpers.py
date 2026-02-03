"""Helper functions for GSO benchmark."""

import os


def get_gso_instance_docker_image(instance_id: str) -> str:
    """Get the Docker image for a GSO instance."""
    # NOTE:
    # - Historically, GSO instance images have been published under slimshetty/gso.
    # - If/when the official repo changes, you can override with GSO_IMAGE_PREFIX.
    #
    # Example override:
    #   export GSO_IMAGE_PREFIX=docker.io/gso-bench/gso
    prefix = (os.environ.get("GSO_IMAGE_PREFIX") or "docker.io/slimshetty/gso").rstrip("/")
    return f"{prefix}:gso.eval.x86_64.{instance_id}".lower()


def get_gso_workspace_dir_name(instance: dict) -> str:
    """Get the workspace directory name for an instance."""
    return instance["repo"].replace("/", "__")
