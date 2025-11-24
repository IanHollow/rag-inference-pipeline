from typing import Any, cast

import orjson
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self


def _orjson_dumps(data: object) -> str:
    return orjson.dumps(data).decode("utf-8")


class BaseJSONModel(BaseModel):
    """
    Base Pydantic model.
    """

    model_config = ConfigDict()

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        **kwargs: object,
    ) -> Self:
        if isinstance(json_data, str):
            json_data = json_data.encode("utf-8")
        obj = orjson.loads(json_data)
        return cls.model_validate(obj, **cast("dict[str, Any]", kwargs))

    def model_dump_json(self, **kwargs: object) -> str:
        dump_kwargs = kwargs.copy()
        if "indent" in dump_kwargs:
            del dump_kwargs["indent"]

        data = self.model_dump(**cast("dict[str, Any]", dump_kwargs))
        return _orjson_dumps(data)
