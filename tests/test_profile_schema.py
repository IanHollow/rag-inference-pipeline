from pydantic import ValidationError
import pytest

from pipeline.config.profile_schema import ComponentConfig, ProfileFile, RouteConfig


def test_valid_profile() -> None:
    profile = ProfileFile(
        name="test_profile",
        components=[ComponentConfig(name="comp1", type="type1", config={"k": "v"})],
        routes=[RouteConfig(target="gateway", prefix="/")],
    )
    assert profile.name == "test_profile"
    assert len(profile.components) == 1
    assert len(profile.routes) == 1


def test_duplicate_prefixes() -> None:
    with pytest.raises(ValidationError):
        ProfileFile(
            name="test",
            routes=[
                RouteConfig(target="gateway", prefix="/"),
                RouteConfig(target="retrieval", prefix="/"),
            ],
        )


def test_invalid_route_target() -> None:
    with pytest.raises(ValidationError):
        RouteConfig(target="invalid", prefix="/")


def test_defaults() -> None:
    profile = ProfileFile(name="defaults")
    assert profile.batch_size is None  # Defaults to None, uses settings.gateway_batch_size
    assert profile.batch_timeout is None  # Defaults to None, uses settings.gateway_batch_timeout_ms
    assert profile.components == []
    assert profile.routes == []
