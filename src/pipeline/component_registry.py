import asyncio
from collections.abc import Callable
import logging

logger = logging.getLogger(__name__)


class ComponentRegistry:
    def __init__(self) -> None:
        self._components: dict[str, object] = {}
        self._lifecycle_hooks: dict[str, dict[str, Callable | None]] = {}
        self._startup_order: list[str] = []

    @property
    def components(self) -> dict[str, object]:
        return self._components

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
            raise ValueError(f"Component {name} already registered")

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
            except Exception as e:
                logger.error("Failed to load component %s: %s", name, e)
                # Cleanup if load fails?
                self.unregister(name)
                raise

    def unregister(self, name: str) -> None:
        if name in self._components:
            del self._components[name]
            del self._lifecycle_hooks[name]
            if name in self._startup_order:
                self._startup_order.remove(name)

    def get(self, name: str) -> object:
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
                except Exception as e:
                    logger.error("Failed to start component %s: %s", name, e)
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
                except Exception as e:
                    logger.error("Failed to stop component %s: %s", name, e)

    def unload_all(self) -> None:
        """Run unload hooks for all components in reverse order."""
        logger.info("Unloading all components...")
        for name in reversed(self._startup_order):
            hook = self._lifecycle_hooks[name].get("unload")
            if hook:
                try:
                    logger.info("Unloading component: %s", name)
                    hook()
                except Exception as e:
                    logger.error("Failed to unload component %s: %s", name, e)
