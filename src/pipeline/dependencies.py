from collections.abc import Callable
from typing import TypeVar

from fastapi import HTTPException, Request, status

from .component_registry import ComponentRegistry

T = TypeVar("T")


def get_registry(request: Request) -> ComponentRegistry:
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Component registry not initialized",
        )
    return registry


def get_component(name: str) -> Callable[[Request], object]:
    def _get_component(request: Request) -> object:
        registry = get_registry(request)

        # Check aliases first
        aliases = getattr(request.app.state, "component_aliases", {})
        resolved_name = aliases.get(name, name)

        component = registry.get(resolved_name)

        return component

    return _get_component
