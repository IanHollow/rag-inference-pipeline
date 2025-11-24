from unittest.mock import patch

import orjson

from pipeline.base_schemas import BaseJSONModel


class JsonPerfModel(BaseJSONModel):
    name: str
    value: int


def test_orjson_usage() -> None:
    """Verify that BaseJSONModel uses orjson for dumping."""
    model = JsonPerfModel(name="test", value=123)

    # We patch orjson.dumps to verify it gets called
    with patch("orjson.dumps", side_effect=orjson.dumps) as mock_orjson_dumps:
        json_output = model.model_dump_json()

        assert mock_orjson_dumps.called
        assert '"name":"test"' in json_output
        assert '"value":123' in json_output


def test_orjson_loading() -> None:
    """Verify that BaseJSONModel uses orjson for loading."""
    json_data = '{"name": "test", "value": 123}'

    with patch("orjson.loads", side_effect=orjson.loads) as mock_orjson_loads:
        model = JsonPerfModel.model_validate_json(json_data)

        assert mock_orjson_loads.called
        assert model.name == "test"
        assert model.value == 123
