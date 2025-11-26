"""
Tests for the base_schemas module.

Tests cover:
- BaseJSONModel with orjson serialization
- model_validate_json with various input types
- model_dump_json output
"""

from pydantic import Field, ValidationError
import pytest

from pipeline.base_schemas import BaseJSONModel


class SampleModel(BaseJSONModel):
    """Sample model for testing BaseJSONModel."""

    name: str
    count: int
    active: bool = True
    tags: list[str] = Field(default_factory=list)


class TestBaseJSONModel:
    """Tests for BaseJSONModel class."""

    def test_model_validate_json_from_string(self) -> None:
        """Test parsing JSON from a string."""
        json_str = '{"name": "test", "count": 42}'

        model = SampleModel.model_validate_json(json_str)

        assert model.name == "test"
        assert model.count == 42
        assert model.active is True  # default value
        assert model.tags == []  # default value

    def test_model_validate_json_from_bytes(self) -> None:
        """Test parsing JSON from bytes."""
        json_bytes = b'{"name": "test", "count": 42}'

        model = SampleModel.model_validate_json(json_bytes)

        assert model.name == "test"
        assert model.count == 42

    def test_model_validate_json_from_bytearray(self) -> None:
        """Test parsing JSON from bytearray."""
        json_bytearray = bytearray(b'{"name": "test", "count": 42}')

        model = SampleModel.model_validate_json(json_bytearray)

        assert model.name == "test"
        assert model.count == 42

    def test_model_validate_json_with_all_fields(self) -> None:
        """Test parsing JSON with all fields specified."""
        json_str = '{"name": "full", "count": 100, "active": false, "tags": ["a", "b"]}'

        model = SampleModel.model_validate_json(json_str)

        assert model.name == "full"
        assert model.count == 100
        assert model.active is False
        assert model.tags == ["a", "b"]

    def test_model_validate_json_unicode(self) -> None:
        """Test parsing JSON with unicode characters."""
        json_str = '{"name": "テスト", "count": 1}'

        model = SampleModel.model_validate_json(json_str)

        assert model.name == "テスト"

    def test_model_dump_json_basic(self) -> None:
        """Test serializing model to JSON string."""
        model = SampleModel(name="test", count=42)

        json_str = model.model_dump_json()

        assert '"name":"test"' in json_str or '"name": "test"' in json_str
        assert '"count":42' in json_str or '"count": 42' in json_str

    def test_model_dump_json_excludes_none_by_default(self) -> None:
        """Test that model_dump_json respects exclude_none if passed."""
        model = SampleModel(name="test", count=42)

        json_str = model.model_dump_json(exclude_none=True)

        # Both fields should be present
        assert "name" in json_str
        assert "count" in json_str

    def test_model_dump_json_with_indent_ignored(self) -> None:
        """Test that indent parameter is handled (ignored for orjson)."""
        model = SampleModel(name="test", count=42)

        # Should not raise even though indent is passed
        json_str = model.model_dump_json(indent=2)

        assert "name" in json_str
        assert "count" in json_str

    def test_roundtrip_serialization(self) -> None:
        """Test that model can be serialized and deserialized without data loss."""
        original = SampleModel(name="roundtrip", count=999, active=False, tags=["x", "y", "z"])

        json_str = original.model_dump_json()
        restored = SampleModel.model_validate_json(json_str)

        assert restored.name == original.name
        assert restored.count == original.count
        assert restored.active == original.active
        assert restored.tags == original.tags

    def test_invalid_json_raises_error(self) -> None:
        """Test that invalid JSON raises an appropriate error."""
        invalid_json = '{"name": "test", count: 42}'  # missing quotes around count

        with pytest.raises(ValueError):  # orjson raises ValueError on decode errors
            SampleModel.model_validate_json(invalid_json)

    def test_missing_required_field_raises_error(self) -> None:
        """Test that missing required field raises validation error."""
        json_str = '{"name": "test"}'  # missing count

        with pytest.raises(ValidationError):
            SampleModel.model_validate_json(json_str)


class NestedModel(BaseJSONModel):
    """Model with nested structure for testing."""

    id: int
    data: SampleModel


class TestNestedModels:
    """Tests for nested BaseJSONModel structures."""

    def test_nested_model_validate_json(self) -> None:
        """Test parsing JSON with nested models."""
        json_str = '{"id": 1, "data": {"name": "nested", "count": 5}}'

        model = NestedModel.model_validate_json(json_str)

        assert model.id == 1
        assert model.data.name == "nested"
        assert model.data.count == 5

    def test_nested_model_dump_json(self) -> None:
        """Test serializing nested models to JSON."""
        inner = SampleModel(name="inner", count=10)
        outer = NestedModel(id=99, data=inner)

        json_str = outer.model_dump_json()

        # Verify nested structure is serialized
        assert "99" in json_str  # id
        assert "inner" in json_str  # nested name
        assert "10" in json_str  # nested count

    def test_nested_roundtrip(self) -> None:
        """Test roundtrip serialization of nested models."""
        inner = SampleModel(name="inner", count=10, tags=["nested"])
        original = NestedModel(id=99, data=inner)

        json_str = original.model_dump_json()
        restored = NestedModel.model_validate_json(json_str)

        assert restored.id == original.id
        assert restored.data.name == original.data.name
        assert restored.data.count == original.data.count
        assert restored.data.tags == original.data.tags
