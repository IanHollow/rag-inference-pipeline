import asyncio
from collections.abc import Callable
import logging


logger = logging.getLogger(__name__)


class ComponentRegistry:
    def __init__(self) -> None:
        self._components: dict[str, object] = {}
        self._aliases: dict[str, str] = {}
        self._lifecycle_hooks: dict[str, dict[str, Callable | None]] = {}
        self._startup_order: list[str] = []

    @property
    def components(self) -> dict[str, object]:
        return self._components

    def register_alias(self, alias: str, name: str) -> None:
        """Register an alias for a component."""
        if alias in self._components:
            msg = f"Alias '{alias}' conflicts with existing component name"
            raise ValueError(msg)
        if alias in self._aliases and self._aliases[alias] != name:
            msg = f"Alias '{alias}' already registered to '{self._aliases[alias]}'"
            raise ValueError(msg)

        self._aliases[alias] = name

    def register(
        self,
        name: str,
        component: object,
        load_hook: Callable | None = None,
        start_hook: Callable | None = None,
        stop_hook: Callable | None = None,
        unload_hook: Callable | None = None,
    ) -> None:
        """
        Register a component and its lifecycle hooks.
        Executes load_hook immediately.
        """
        if name in self._components:
            msg = f"Component {name} already registered"
            raise ValueError(msg)

        self._components[name] = component
        self._lifecycle_hooks[name] = {
            "load": load_hook,
            "start": start_hook,
            "stop": stop_hook,
            "unload": unload_hook,
        }
        self._startup_order.append(name)

        # Execute load hook immediately
        if load_hook:
            try:
                logger.info("Loading component: %s", name)
                load_hook()
            except Exception:
                logger.exception("Failed to load component %s", name)
                # Cleanup if load fails?
                self.unregister(name)
                raise

    def unregister(self, name: str) -> None:
        if name in self._components:
            del self._components[name]
            del self._lifecycle_hooks[name]
            if name in self._startup_order:
                self._startup_order.remove(name)

            # Remove aliases pointing to this component
            aliases_to_remove = [k for k, v in self._aliases.items() if v == name]
            for k in aliases_to_remove:
                del self._aliases[k]

    def get(self, name: str) -> object:
        if name in self._aliases:
            name = self._aliases[name]
        return self._components.get(name)

    async def start_all(self) -> None:
        """Run start hooks for all components."""
        logger.info("Starting all components...")
        for name in self._startup_order:
            hook = self._lifecycle_hooks[name].get("start")
            if hook:
                try:
                    logger.info("Starting component: %s", name)
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception:
                    logger.exception("Failed to start component %s", name)
                    raise

    async def stop_all(self) -> None:
        """Run stop hooks for all components in reverse order."""
        logger.info("Stopping all components...")
        for name in reversed(self._startup_order):
            hook = self._lifecycle_hooks[name].get("stop")
            if hook:
                try:
                    logger.info("Stopping component: %s", name)
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception:
                    logger.exception("Failed to stop component %s", name)

    def unload_all(self) -> None:
        """Run unload hooks for all components in reverse order."""
        logger.info("Unloading all components...")
        for name in reversed(self._startup_order):
            hook = self._lifecycle_hooks[name].get("unload")
            if hook:
                try:
                    logger.info("Unloading component: %s", name)
                    hook()
                except Exception:
                    logger.exception("Failed to unload component %s", name)
