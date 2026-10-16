
"""Tests for the ExperimentConfig class."""

import pytest
import yaml

from pneumonia.config import ExperimentConfig


def test_default_config_loads():
    """The default configuration should load successfully."""
    config = ExperimentConfig()

    assert config is not None
    assert isinstance(config.to_dict(), dict)


def test_default_config_contains_expected_sections():
    """The configuration should contain its main sections."""
    config = ExperimentConfig()
    data = config.to_dict()

    expected_sections = [
        "experiment",
        "training",
        "model",
        "data",
        "evaluation",
        "output",
    ]

    for section in expected_sections:
        assert section in data


def test_get_value_returns_existing_value():
    """get_value should retrieve a configured value."""
    config = ExperimentConfig()

    epochs = config.get_value("training", "num_epochs")

    assert epochs == 50


def test_get_value_returns_default_for_missing_value():
    """get_value should return the fallback for a missing key."""
    config = ExperimentConfig()

    result = config.get_value(
        "training",
        "missing_setting",
        default="fallback",
    )

    assert result == "fallback"


def test_overrides_update_configuration():
    """Overrides should replace the corresponding default values."""
    config = ExperimentConfig(
        overrides={
            "training": {
                "num_epochs": 5,
                "batch_size": 8,
            }
        }
    )

    assert config.get_value("training", "num_epochs") == 5
    assert config.get_value("training", "batch_size") == 8


def test_overrides_preserve_unmodified_values():
    """Overriding one setting should preserve other defaults."""
    config = ExperimentConfig(
        overrides={
            "training": {
                "num_epochs": 5,
            }
        }
    )

    assert config.get_value("training", "num_epochs") == 5
    assert config.get_value("training", "batch_size") == 32


def test_invalid_num_epochs_raises_error():
    """Training epochs must be positive."""
    with pytest.raises((ValueError, TypeError)):
        ExperimentConfig(
            overrides={
                "training": {
                    "num_epochs": 0,
                }
            }
        )


def test_invalid_batch_size_raises_error():
    """Batch size must be positive."""
    with pytest.raises((ValueError, TypeError)):
        ExperimentConfig(
            overrides={
                "training": {
                    "batch_size": 0,
                }
            }
        )


def test_invalid_learning_rate_raises_error():
    """Learning rate must be positive."""
    with pytest.raises((ValueError, TypeError)):
        ExperimentConfig(
            overrides={
                "training": {
                    "learning_rate": -0.001,
                }
            }
        )


def test_yaml_config_loads(tmp_path):
    """A valid YAML configuration file should load."""
    config_path = tmp_path / "test_config.yaml"

    yaml_data = {
        "experiment": {
            "name": "TestExperiment",
        },
        "training": {
            "num_epochs": 3,
            "batch_size": 4,
        },
    }

    config_path.write_text(
        yaml.safe_dump(yaml_data),
        encoding="utf-8",
    )

    config = ExperimentConfig(config_path=config_path)

    assert config.get_value("experiment", "name") == "TestExperiment"
    assert config.get_value("training", "num_epochs") == 3
    assert config.get_value("training", "batch_size") == 4


def test_save_and_reload_config(tmp_path):
    """A saved configuration should be loadable again."""
    config = ExperimentConfig(
        overrides={
            "experiment": {
                "name": "SaveReloadTest",
            },
            "training": {
                "num_epochs": 7,
            },
        }
    )

    output_path = tmp_path / "saved_config.yaml"
    config.save(output_path)

    assert output_path.exists()

    reloaded = ExperimentConfig(config_path=output_path)

    assert reloaded.get_value("experiment", "name") == "SaveReloadTest"
    assert reloaded.get_value("training", "num_epochs") == 7


def test_to_dict_returns_dictionary():
    """to_dict should return the configuration as a dictionary."""
    config = ExperimentConfig()

    result = config.to_dict()

    assert isinstance(result, dict)
    assert "training" in result