"""
Tests for the ComponentRegistry class.

Tests cover:
- Component registration and retrieval
- Alias management
- Lifecycle hooks (load, start, stop, unload)
- Error handling for duplicate registrations and invalid aliases
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.component_registry import ComponentRegistry


class TestComponentRegistration:
    """Tests for basic component registration and retrieval."""

    def test_register_component(self) -> None:
        """Test registering a component."""
        registry = ComponentRegistry()
        component = MagicMock()

        registry.register("test_component", component)

        assert "test_component" in registry.components
        assert registry.get("test_component") == component

    def test_register_duplicate_raises_error(self) -> None:
        """Test that registering a duplicate component raises ValueError."""
        registry = ComponentRegistry()
        component = MagicMock()

        registry.register("test_component", component)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test_component", MagicMock())

    def test_get_nonexistent_component(self) -> None:
        """Test that getting a nonexistent component returns None."""
        registry = ComponentRegistry()

        result = registry.get("nonexistent")

        assert result is None

    def test_unregister_component(self) -> None:
        """Test unregistering a component."""
        registry = ComponentRegistry()
        component = MagicMock()
        registry.register("test_component", component)

        registry.unregister("test_component")

        assert "test_component" not in registry.components
        assert registry.get("test_component") is None

    def test_unregister_nonexistent_component_no_error(self) -> None:
        """Test that unregistering a nonexistent component doesn't raise error."""
        registry = ComponentRegistry()

        # Should not raise
        registry.unregister("nonexistent")

    def test_components_property(self) -> None:
        """Test that components property returns the correct dictionary."""
        registry = ComponentRegistry()
        component1 = MagicMock()
        component2 = MagicMock()

        registry.register("comp1", component1)
        registry.register("comp2", component2)

        components = registry.components
        assert len(components) == 2
        assert components["comp1"] == component1
        assert components["comp2"] == component2


class TestAliases:
    """Tests for alias management."""

    def test_register_alias(self) -> None:
        """Test registering an alias for a component."""
        registry = ComponentRegistry()
        component = MagicMock()
        registry.register("actual_name", component)

        registry.register_alias("alias_name", "actual_name")

        assert registry.get("alias_name") == component

    def test_alias_conflicts_with_component_name(self) -> None:
        """Test that alias conflicting with component name raises error."""
        registry = ComponentRegistry()
        registry.register("existing_name", MagicMock())

        with pytest.raises(ValueError, match="conflicts with existing component"):
            registry.register_alias("existing_name", "other_name")

    def test_alias_already_registered_to_different_component(self) -> None:
        """Test that re-registering alias to different component raises error."""
        registry = ComponentRegistry()
        registry.register("comp1", MagicMock())
        registry.register("comp2", MagicMock())
        registry.register_alias("alias", "comp1")

        with pytest.raises(ValueError, match="already registered"):
            registry.register_alias("alias", "comp2")

    def test_alias_same_target_no_error(self) -> None:
        """Test that registering same alias to same target doesn't raise."""
        registry = ComponentRegistry()
        registry.register("comp1", MagicMock())
        registry.register_alias("alias", "comp1")

        # Should not raise
        registry.register_alias("alias", "comp1")

    def test_unregister_removes_aliases(self) -> None:
        """Test that unregistering a component removes its aliases."""
        registry = ComponentRegistry()
        component = MagicMock()
        registry.register("comp1", component)
        registry.register_alias("alias1", "comp1")
        registry.register_alias("alias2", "comp1")

        registry.unregister("comp1")

        # Aliases should no longer resolve
        assert registry.get("alias1") is None
        assert registry.get("alias2") is None


class TestLifecycleHooks:
    """Tests for component lifecycle hooks."""

    def test_load_hook_called_on_register(self) -> None:
        """Test that load hook is called immediately on registration."""
        registry = ComponentRegistry()
        component = MagicMock()
        load_hook = MagicMock()

        registry.register("comp", component, load_hook=load_hook)

        load_hook.assert_called_once()

    def test_load_hook_failure_unregisters_component(self) -> None:
        """Test that failed load hook unregisters the component."""
        registry = ComponentRegistry()
        component = MagicMock()
        load_hook = MagicMock(side_effect=Exception("Load failed"))

        with pytest.raises(Exception, match="Load failed"):
            registry.register("comp", component, load_hook=load_hook)

        assert "comp" not in registry.components

    @pytest.mark.asyncio
    async def test_start_all_calls_start_hooks(self) -> None:
        """Test that start_all calls start hooks for all components."""
        registry = ComponentRegistry()
        start_hook1 = MagicMock()
        start_hook2 = MagicMock()

        registry.register("comp1", MagicMock(), start_hook=start_hook1)
        registry.register("comp2", MagicMock(), start_hook=start_hook2)

        await registry.start_all()

        start_hook1.assert_called_once()
        start_hook2.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_all_with_async_hooks(self) -> None:
        """Test that start_all handles async start hooks."""
        registry = ComponentRegistry()
        async_hook = AsyncMock()

        registry.register("comp1", MagicMock(), start_hook=async_hook)

        await registry.start_all()

        async_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_all_respects_order(self) -> None:
        """Test that start_all respects registration order."""
        registry = ComponentRegistry()
        call_order: list[str] = []

        def make_hook(name: str) -> MagicMock:
            def hook() -> None:
                call_order.append(name)

            return MagicMock(side_effect=hook)

        registry.register("comp1", MagicMock(), start_hook=make_hook("comp1"))
        registry.register("comp2", MagicMock(), start_hook=make_hook("comp2"))
        registry.register("comp3", MagicMock(), start_hook=make_hook("comp3"))

        await registry.start_all()

        assert call_order == ["comp1", "comp2", "comp3"]

    @pytest.mark.asyncio
    async def test_stop_all_calls_stop_hooks_reverse_order(self) -> None:
        """Test that stop_all calls stop hooks in reverse order."""
        registry = ComponentRegistry()
        call_order: list[str] = []

        def make_hook(name: str) -> MagicMock:
            def hook() -> None:
                call_order.append(name)

            return MagicMock(side_effect=hook)

        registry.register("comp1", MagicMock(), stop_hook=make_hook("comp1"))
        registry.register("comp2", MagicMock(), stop_hook=make_hook("comp2"))
        registry.register("comp3", MagicMock(), stop_hook=make_hook("comp3"))

        await registry.stop_all()

        assert call_order == ["comp3", "comp2", "comp1"]

    @pytest.mark.asyncio
    async def test_stop_all_with_async_hooks(self) -> None:
        """Test that stop_all handles async stop hooks."""
        registry = ComponentRegistry()
        async_hook = AsyncMock()

        registry.register("comp1", MagicMock(), stop_hook=async_hook)

        await registry.stop_all()

        async_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_all_continues_on_error(self) -> None:
        """Test that stop_all continues even if a hook fails."""
        registry = ComponentRegistry()
        failing_hook = MagicMock(side_effect=Exception("Stop failed"))
        success_hook = MagicMock()

        registry.register("comp1", MagicMock(), stop_hook=success_hook)
        registry.register("comp2", MagicMock(), stop_hook=failing_hook)

        # Should not raise
        await registry.stop_all()

        # Both hooks should have been attempted
        failing_hook.assert_called_once()
        success_hook.assert_called_once()

    def test_unload_all_calls_unload_hooks_reverse_order(self) -> None:
        """Test that unload_all calls unload hooks in reverse order."""
        registry = ComponentRegistry()
        call_order: list[str] = []

        def make_hook(name: str) -> MagicMock:
            def hook() -> None:
                call_order.append(name)

            return MagicMock(side_effect=hook)

        registry.register("comp1", MagicMock(), unload_hook=make_hook("comp1"))
        registry.register("comp2", MagicMock(), unload_hook=make_hook("comp2"))
        registry.register("comp3", MagicMock(), unload_hook=make_hook("comp3"))

        registry.unload_all()

        assert call_order == ["comp3", "comp2", "comp1"]

    def test_unload_all_continues_on_error(self) -> None:
        """Test that unload_all continues even if a hook fails."""
        registry = ComponentRegistry()
        failing_hook = MagicMock(side_effect=Exception("Unload failed"))
        success_hook = MagicMock()

        registry.register("comp1", MagicMock(), unload_hook=success_hook)
        registry.register("comp2", MagicMock(), unload_hook=failing_hook)

        # Should not raise
        registry.unload_all()

        # Both hooks should have been attempted
        failing_hook.assert_called_once()
        success_hook.assert_called_once()


class TestStartupOrder:
    """Tests for startup order tracking."""

    def test_startup_order_maintained(self) -> None:
        """Test that startup order is maintained after registration."""
        registry = ComponentRegistry()

        registry.register("comp1", MagicMock())
        registry.register("comp2", MagicMock())
        registry.register("comp3", MagicMock())

        assert registry._startup_order == ["comp1", "comp2", "comp3"]

    def test_unregister_updates_startup_order(self) -> None:
        """Test that unregister updates the startup order."""
        registry = ComponentRegistry()

        registry.register("comp1", MagicMock())
        registry.register("comp2", MagicMock())
        registry.register("comp3", MagicMock())

        registry.unregister("comp2")

        assert registry._startup_order == ["comp1", "comp3"]
