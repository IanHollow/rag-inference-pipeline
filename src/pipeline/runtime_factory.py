from collections.abc import AsyncIterator, Callable
import contextlib
from contextlib import AbstractAsyncContextManager
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from prometheus_client import generate_latest
import yaml

from .component_factory import create_component
from .component_registry import ComponentRegistry
from .config import PipelineSettings
from .config.profile_schema import ComponentConfig, ProfileFile
from .enums import ComponentType
from .middleware import CompressionMiddleware

# Import routers
from .services.gateway.api import router as gateway_router
from .services.generation.api import (
    router as generation_router,
    start_generation_executor,
    stop_generation_executor,
)
from .services.retrieval.api import (
    router as retrieval_router,
    start_retrieval_executor,
    stop_retrieval_executor,
)
from .telemetry import instrument_fastapi_app


logger = logging.getLogger(__name__)


def _validate_alias_unique(
    alias: str, component_name: str, existing_aliases: dict[str, str]
) -> None:
    """Validate that an alias is unique, raising ValueError if duplicate."""
    if alias in existing_aliases:
        msg = f"Duplicate alias '{alias}' defined in component '{component_name}'"
        raise ValueError(msg)


def _get_default_alias_for_type(ctype: ComponentType | str) -> str | None:
    """Get the default alias for a component type."""
    alias_map: dict[ComponentType | str, str] = {
        ComponentType.EMBEDDING: "embedding_generator",
        "embedding_generator": "embedding_generator",
        ComponentType.FAISS: "faiss_store",
        "faiss_store": "faiss_store",
        ComponentType.DOCUMENT_STORE: "document_store",
        ComponentType.RERANKER: "reranker",
        ComponentType.LLM: "llm_generator",
        "llm_generator": "llm_generator",
        ComponentType.SENTIMENT: "sentiment_analyzer",
        "sentiment_analyzer": "sentiment_analyzer",
        ComponentType.TOXICITY: "toxicity_filter",
        "toxicity_filter": "toxicity_filter",
        ComponentType.GATEWAY: "orchestrator",
        "orchestrator": "orchestrator",
    }
    return alias_map.get(ctype)


def load_role_profile(settings: PipelineSettings) -> ProfileFile:
    """
    Load the role profile based on settings.
    """
    path = None

    # 1. Load from file if specified
    if settings.role_profile_override_path:
        path = Path(settings.role_profile_override_path)

    # 2. Load by name
    elif settings.pipeline_role_profile:
        # Look in configs/ directory relative to project root
        # This file is src/pipeline/runtime_factory.py
        base_dir = Path(__file__).resolve().parent.parent.parent
        path = base_dir / "configs" / f"{settings.pipeline_role_profile}.yaml"
        if not path.exists():
            path = base_dir / "configs" / f"{settings.pipeline_role_profile}.yml"

    # 3. Fallback: Derive from node number if not set (in case validator failed)
    if not path and not settings.role_profile_override_path:
        profile_name = ""
        if settings.node_number == 0:
            profile_name = "gateway_with_embedding"
        elif settings.node_number == 1:
            profile_name = "retrieval"
        elif settings.node_number == 2:
            profile_name = "generation"

        if profile_name:
            logger.info(
                "Deriving role profile from node number %d: %s", settings.node_number, profile_name
            )
            base_dir = Path(__file__).resolve().parent.parent.parent
            path = base_dir / "configs" / f"{profile_name}.yaml"
            if not path.exists():
                path = base_dir / "configs" / f"{profile_name}.yml"

    if not path or not path.exists():
        msg = (
            f"No valid role profile found. "
            f"Please set PIPELINE_ROLE_PROFILE to a valid profile name (e.g. 'gateway') "
            f"or ROLE_PROFILE_OVERRIDE_PATH to a YAML file. "
            f"Checked path: {path}"
        )
        raise ValueError(msg)

    logger.info("Loading role profile from file: %s", path)
    try:
        with Path(path).open() as f:
            data = yaml.safe_load(f)
        # Validate with Pydantic
        return ProfileFile(**data)
    except Exception as e:
        logger.exception("Failed to parse role profile from file")
        msg = f"Invalid role profile file: {e}"
        raise ValueError(msg) from e


def _create_base_app(
    settings: PipelineSettings,
    profile: ProfileFile,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]],
) -> FastAPI:
    """Create the base FastAPI app with middleware configured."""
    app = FastAPI(
        title=f"ML Pipeline Node {settings.node_number}",
        description=f"Role: {profile.name} - {profile.description}",
        version="0.1.0",
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
    )

    app.router.default_response_class = ORJSONResponse

    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add Custom Compression (runs before GZip)
    app.add_middleware(CompressionMiddleware)

    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Instrument with telemetry
    instrument_fastapi_app(app)

    return app


def _register_component_aliases(
    app: FastAPI,
    registry: ComponentRegistry,
    component_config: ComponentConfig,
    ctype: ComponentType | str,
) -> None:
    """Register aliases for a component."""
    # Register explicit aliases from component config
    for alias in component_config.aliases:
        _validate_alias_unique(alias, component_config.name, app.state.component_aliases)
        app.state.component_aliases[alias] = component_config.name
        registry.register_alias(alias, component_config.name)

    # Register default alias based on type
    default_alias = _get_default_alias_for_type(ctype)

    if (
        default_alias
        and default_alias not in app.state.component_aliases
        and default_alias != component_config.name
    ):
        app.state.component_aliases[default_alias] = component_config.name
        registry.register_alias(default_alias, component_config.name)


def _initialize_component(
    app: FastAPI,
    registry: ComponentRegistry,
    settings: PipelineSettings,
    profile: ProfileFile,
    component_config: ComponentConfig,
) -> None:
    """Initialize and register a single component."""
    logger.info("Creating component: %s (%s)", component_config.name, component_config.type)

    # Inject profile-level settings for Gateway
    if component_config.type == ComponentType.GATEWAY.value:
        if profile.batch_size is not None:
            component_config.config["batch_size"] = profile.batch_size
        if profile.batch_timeout is not None:
            component_config.config["batch_timeout"] = profile.batch_timeout

    # Convert string type to Enum if needed
    ctype: ComponentType | str = component_config.type
    with contextlib.suppress(ValueError):
        ctype = ComponentType(component_config.type)

    component = create_component(ctype, settings, component_config.config)

    # Get lifecycle hooks
    load_hook = getattr(component, "load", None)
    start_hook = getattr(component, "start", None)
    stop_hook = getattr(component, "stop", None)
    unload_hook = getattr(component, "unload", None)

    # Support close_all as stop if stop is missing
    if not stop_hook and hasattr(component, "close_all"):
        stop_hook = getattr(component, "close_all", None)

    registry.register(
        name=component_config.name,
        component=component,
        load_hook=load_hook,
        start_hook=start_hook,
        stop_hook=stop_hook,
        unload_hook=unload_hook,
    )

    # Mount component to app state for backward compatibility
    setattr(app.state, component_config.name, component)

    _register_component_aliases(app, registry, component_config, ctype)


def _mount_routers(
    app: FastAPI,
    registry: ComponentRegistry,
    profile: ProfileFile,
) -> None:
    """Mount routers based on profile routes."""
    router_map = {
        "gateway": gateway_router,
        "retrieval": retrieval_router,
        "generation": generation_router,
    }

    for route_config in profile.routes:
        target = route_config.target
        prefix = route_config.prefix

        if target not in router_map:
            logger.warning("Unknown route target: %s", target)
            continue

        router = router_map[target]
        logger.info("Mounting router for target '%s' at prefix '%s'", target, prefix)
        app.include_router(router, prefix=prefix if prefix != "/" else "")

        # Store component aliases from route config
        if route_config.component_aliases:
            for alias, target_name in route_config.component_aliases.items():
                if (
                    alias in app.state.component_aliases
                    and app.state.component_aliases[alias] != target_name
                ):
                    msg = f"Duplicate alias '{alias}' defined in route '{prefix}'"
                    raise ValueError(msg)
                app.state.component_aliases[alias] = target_name
                registry.register_alias(alias, target_name)


def _register_endpoints(
    app: FastAPI,
    settings: PipelineSettings,
    profile: ProfileFile,
) -> None:
    """Register health check and metrics endpoints."""

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        registry = getattr(app.state, "registry", None)
        components_status = {}
        overall_status = "healthy"

        if registry:
            for name, component in registry.components.items():
                is_loaded = getattr(component, "is_loaded", True)
                status_str = "ready" if is_loaded else "initializing"
                if not is_loaded:
                    overall_status = "initializing"
                components_status[name] = {"status": status_str, "type": type(component).__name__}

        return {
            "status": overall_status,
            "node": settings.node_number,
            "role": profile.name,
            "components": components_status,
        }

    @app.get("/metrics", response_class=Response)
    @app.head("/metrics", response_class=Response)
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )


def create_app_from_profile(settings: PipelineSettings) -> FastAPI:
    """
    Create a FastAPI application based on the resolved role profile.
    """
    profile = load_role_profile(settings)
    registry = ComponentRegistry()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info("Application startup: %s", profile.name)
        await registry.start_all()

        has_retrieval = any(r.target == "retrieval" for r in profile.routes)
        has_generation = any(r.target == "generation" for r in profile.routes)

        if has_retrieval:
            await start_retrieval_executor(registry)
        if has_generation:
            await start_generation_executor(registry)

        yield

        if has_generation:
            await stop_generation_executor()
        if has_retrieval:
            await stop_retrieval_executor()

        logger.info("Application shutdown")
        await registry.stop_all()
        registry.unload_all()

    app = _create_base_app(settings, profile, lifespan)

    # Initialize Component Registry
    app.state.registry = registry
    app.state.component_aliases = {}

    # Initialize components
    logger.info("Initializing components for profile: %s", profile.name)
    try:
        for component_config in profile.components:
            _initialize_component(app, registry, settings, profile, component_config)
    except Exception:
        logger.exception("Failed to create components")
        raise

    _mount_routers(app, registry, profile)
    _register_endpoints(app, settings, profile)

    return app
