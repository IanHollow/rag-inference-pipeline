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
    batch_size: int = 32
    batch_timeout: float = 0.1
    components: list[ComponentConfig] = Field(default_factory=list)
    routes: list[RouteConfig] = Field(default_factory=list)

    @field_validator("routes")
    @classmethod
    def check_duplicate_prefixes(cls, v: list[RouteConfig]) -> list[RouteConfig]:
        prefixes = [route.prefix for route in v]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("Duplicate prefixes found in routes")
        return v

    @model_validator(mode="after")
    def check_aliases_point_to_components(self) -> "ProfileFile":
        component_names = {c.name for c in self.components}
        for route in self.routes:
            for alias, target_name in route.component_aliases.items():
                if target_name not in component_names:
                    raise ValueError(
                        f"Route alias '{alias}' points to unknown component '{target_name}'"
                    )
        return self
