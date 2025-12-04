from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ComponentConfig(BaseModel):
    name: str
    type: str
    config: dict[str, Any] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)


class RouteConfig(BaseModel):
    target: Literal["gateway", "retrieval", "generation"]
    prefix: str = "/"
    component_aliases: dict[str, str] = Field(default_factory=dict)


class ProfileFile(BaseModel):
    name: str
    description: str = ""
    batch_size: int | None = None  # If None, use settings.gateway_batch_size
    batch_timeout: float | None = None  # If None, use settings.gateway_batch_timeout_ms
    components: list[ComponentConfig] = Field(default_factory=list)
    routes: list[RouteConfig] = Field(default_factory=list)

    @field_validator("routes")
    @classmethod
    def check_duplicate_prefixes(cls, v: list[RouteConfig]) -> list[RouteConfig]:
        prefixes = [route.prefix for route in v]
        if len(prefixes) != len(set(prefixes)):
            msg = "Duplicate prefixes found in routes"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def check_aliases_point_to_components(self) -> "ProfileFile":
        component_names = {c.name for c in self.components}
        for route in self.routes:
            for alias, target_name in route.component_aliases.items():
                if target_name not in component_names:
                    msg = f"Route alias '{alias}' points to unknown component '{target_name}'"
                    raise ValueError(msg)
        return self
