import argparse
import datetime
import glob
import importlib
import os
import types
import typing
from pathlib import Path
from typing import Any

from molmo_spaces.data_generation.config_registry import get_config_class
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

"""
Main script entry for data generation.

To run:
- Set terminal at molmo-spaces root directory
- For MacOS, set the following environment variables:
  - export PYTHONPATH="${PYTHONPATH}:."
  - export MUJOCO_GL=egl
  - export PYOPENGL_PLATFORM=egl
- Example commands:
  - python -m molmo_spaces.data_generation.main DoorOpeningDebugConfig
  - python -m molmo_spaces.data_generation.main DoorOpeningDataGenConfig
- You may also pass additional experiment config arguments for your experiment config class as command line arguments.
- You can also set JAX_COMPILATION_CACHE_DIR to cache compiled jax functions between runs, which could speed up initialization.

Config classes are auto-discovered from the config_registry. To add a new config:
1. Create your config class in any file under config/
2. Add @register_config("YourConfigName") decorator to register it
3. Use the registered name as the command line argument
"""


def get_args():
    parser = argparse.ArgumentParser(
        description="MolmoSpaces data generation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "exp_config_cls",
        type=str,
        help="Name of the experiment config class to use (e.g., FrankaPickDroidDataGenConfig), "
        "optionally with the module name prepended with a colon (e.g. molmo_spaces.data_generation.config.object_manipulation_datagen_configs:FrankaPickDroidDataGenConfig). "
        "If the module is specified, only that module will be imported to populate the registry. Otherwise, all config files will be imported.",
    )
    # Any remaining --key value (or --key=value) pairs are forwarded as overrides
    # onto the resolved experiment config (see _apply_cli_overrides). This lets
    # callers tweak fields like num_workers / task_horizon / output_dir / data_split
    # without having to edit the config class.
    return parser.parse_known_args()


def _parse_overrides(extras: list[str]) -> dict[str, str]:
    """Parse leftover argv tokens like ['--num_workers', '5', '--output_dir=foo']."""
    overrides: dict[str, str] = {}
    i = 0
    while i < len(extras):
        token = extras[i]
        if not token.startswith("--"):
            raise SystemExit(f"Unexpected positional argument: {token!r}")
        key = token[2:]
        if "=" in key:
            key, value = key.split("=", 1)
        else:
            i += 1
            if i >= len(extras):
                raise SystemExit(f"Missing value for argument: --{key}")
            value = extras[i]
        overrides[key] = value
        i += 1
    return overrides


def _coerce_value(annotation: Any, raw: str) -> Any:
    """Best-effort coerce a string CLI value to the field's annotated type."""
    # Unwrap Optional[X] / Union[X, None] / `X | None`. PEP 604 unions resolve to
    # types.UnionType (not typing.Union), so we accept both origins.
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        if non_none:
            annotation = non_none[0]
    if annotation is bool:
        return raw.lower() in ("1", "true", "yes", "on", "y", "t")
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    if annotation is Path:
        return Path(raw)
    return raw  # fall back to raw string; pydantic will coerce on assignment


def _apply_cli_overrides(exp_config: Any, overrides: dict[str, str]) -> None:
    fields = type(exp_config).model_fields
    for key, raw in overrides.items():
        if key not in fields:
            raise SystemExit(
                f"Unknown config field: --{key}. Available top-level fields: "
                f"{sorted(fields)}"
            )
        coerced = _coerce_value(fields[key].annotation, raw)
        setattr(exp_config, key, coerced)


def auto_import_configs() -> None:
    """Auto-import all config files so they register themselves"""
    # Get the config directory path
    current_dir = os.path.dirname(__file__)
    config_dir = os.path.join(current_dir, "config")

    if not os.path.exists(config_dir):
        print(f"Warning: Config directory not found: {config_dir}")
        return

    # Import all .py files in the config directory
    config_files = glob.glob(os.path.join(config_dir, "*.py"))

    for config_path in config_files:
        # Skip __init__.py
        if config_path.endswith("__init__.py"):
            continue

        # Load the module with the full module path for proper pickling
        module_filename = os.path.splitext(os.path.basename(config_path))[0]
        full_module_name = f"molmo_spaces.data_generation.config.{module_filename}"

        # Use standard import instead of spec_from_file_location
        # This ensures the module has the correct __name__ for pickling
        try:
            importlib.import_module(full_module_name)
        except Exception as e:
            print(f"Warning: Could not load config from {full_module_name}: {e}")
            continue


def main() -> None:
    args, extras = get_args()
    exp_config_cls = args.exp_config_cls
    overrides = _parse_overrides(extras)

    # np.random.seed(42)

    if (
        ":" in exp_config_cls
    ):  # if the module is specified, import it and use the class from that module
        exp_config_module, exp_config_cls = exp_config_cls.split(":")
        importlib.import_module(exp_config_module)
    else:  # otherwise, auto-import all config files to populate the registry
        auto_import_configs()

    # Get the config class from the registry
    ExpConfigClass = get_config_class(exp_config_cls)
    # Construct with config defaults; CLI overrides are then applied below.
    exp_config = ExpConfigClass()
    _apply_cli_overrides(exp_config, overrides)
    if "allocate_unique_house_subdirs" not in overrides:
        exp_config.allocate_unique_house_subdirs = True

    # Optional: Modify the config parameters here if needed
    # Eg. for hyperparamter sweeps etc.

    # Generate unique run name
    exp_config_name = exp_config_cls  # Use the class name directly

    # Determine output directory structure
    # For local debugging (non-shared paths), add timestamp to avoid collisions
    # For production (datagen output targets shared filesystem paths), use simple structure
    is_shared_fs_path = "/mnt/shared" in str(
        exp_config.output_dir
    )  # or whatever your mount point is

    if is_shared_fs_path:
        # Production: simple structure without timestamp
        exp_config.output_dir = exp_config.output_dir / exp_config_name
    else:
        # Local debug: add timestamp to avoid collisions
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_config.output_dir = exp_config.output_dir / exp_config_name / timestamp

    os.makedirs(exp_config.output_dir, exist_ok=True)

    # Initialize wandb if enabled - with auto run name
    if exp_config.use_wandb:
        import wandb

        if exp_config.wandb_name is None:
            # Generate timestamp for wandb run name
            wandb_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_config.wandb_name = f"{exp_config_name}_{wandb_timestamp}"
        wandb.init(
            project=exp_config.wandb_project, name=exp_config.wandb_name, config=vars(exp_config)
        )

    exp_config.save_config()

    # Create rollout runner with the set config parameters
    runner = ParallelRolloutRunner(exp_config)

    success_count, total_count = runner.run()
    print(f"Success count: {success_count}, Total count: {total_count}")

    # Close wandb run
    if exp_config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
