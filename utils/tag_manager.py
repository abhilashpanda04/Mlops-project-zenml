"""
Utility scripts for managing and querying tagged ZenML resources.

Demonstrates:
- Finding pipelines, artifacts, and models by tags
- Advanced tag filtering (startswith, contains)
- Finding orphaned (untagged) resources
"""

from zenml.client import Client
from rich import print as rprint

from tag_registry import Environment, ArtifactType


def find_pipeline_runs_by_tag(tags: list[str]) -> None:
    """Find and display pipeline runs matching given tags."""
    client = Client()
    runs = client.list_pipeline_runs(tags=tags)

    rprint(f"\n[bold]Found {len(runs.items)} pipeline runs with tags {tags}:[/bold]")
    for run in runs.items:
        tag_names = [tag.name for tag in run.tags]
        rprint(f"  - {run.name} (status: {run.status}, tags: {', '.join(tag_names)})")


def find_artifacts_by_tag(tags: list[str]) -> None:
    """Find and display artifact versions matching given tags."""
    client = Client()
    artifacts = client.list_artifact_versions(tags=tags)

    rprint(f"\n[bold]Found {len(artifacts.items)} artifacts with tags {tags}:[/bold]")
    for artifact in artifacts.items:
        tag_names = [tag.name for tag in artifact.tags]
        rprint(f"  - {artifact.name} v{artifact.version} (type: {artifact.type}, tags: {', '.join(tag_names)})")


def find_models_by_tag(tags: list[str]) -> None:
    """Find and display models matching given tags."""
    client = Client()
    models = client.list_models(tags=tags)

    rprint(f"\n[bold]Found {len(models.items)} models with tags {tags}:[/bold]")
    for model in models.items:
        tag_names = [tag.name for tag in model.tags]
        rprint(f"  - {model.name} (tags: {', '.join(tag_names)})")


def find_high_performance_models() -> None:
    """Find models tagged with high performance metrics using prefix filtering."""
    client = Client()
    high_r2_artifacts = client.list_artifact_versions(
        tags=["startswith:performance-high"]
    )

    rprint(f"\n[bold]High performance model artifacts:[/bold]")
    for artifact in high_r2_artifacts.items:
        tag_names = [tag.name for tag in artifact.tags]
        rprint(f"  - {artifact.name} v{artifact.version} (tags: {', '.join(tag_names)})")


def find_untagged_resources() -> None:
    """Find resources without environment tags (orphaned resources)."""
    client = Client()

    env_tags = [e.value for e in Environment]

    # Check pipeline runs without environment tags
    all_runs = client.list_pipeline_runs().items
    untagged_runs = []

    for run in all_runs:
        run_tags = [tag.name for tag in run.tags]
        if not any(env_tag in run_tags for env_tag in env_tags):
            untagged_runs.append(run)

    rprint(f"\n[bold yellow]⚠ Found {len(untagged_runs)} pipeline runs without environment tags:[/bold yellow]")
    for run in untagged_runs:
        rprint(f"  - {run.name} (tags: {[tag.name for tag in run.tags]})")

    # Check artifacts without type tags
    type_tags = [t.value for t in ArtifactType]
    all_artifacts = client.list_artifact_versions().items
    untagged_artifacts = []

    for artifact in all_artifacts:
        art_tags = [tag.name for tag in artifact.tags]
        if not any(type_tag in art_tags for type_tag in type_tags):
            untagged_artifacts.append(artifact)

    rprint(f"[bold yellow]⚠ Found {len(untagged_artifacts)} artifacts without type tags[/bold yellow]")


if __name__ == "__main__":
    rprint("[bold green]═══ ZenML Resource Tag Report ═══[/bold green]\n")

    # Find training pipeline runs
    find_pipeline_runs_by_tag(["pipeline-training"])

    # Find raw data artifacts
    find_artifacts_by_tag([ArtifactType.RAW.value])

    # Find processed artifacts
    find_artifacts_by_tag([ArtifactType.PROCESSED.value])

    # Find model artifacts
    find_artifacts_by_tag([ArtifactType.MODEL.value])

    # Find high-performance models
    find_high_performance_models()

    # Find orphaned resources
    find_untagged_resources()

    rprint("\n[bold green]═══ Report Complete ═══[/bold green]")
