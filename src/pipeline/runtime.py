"""
Runtime entry point for the distributed ML pipeline.

This module is invoked via `python -m pipeline.runtime`
"""

import logging
import signal
import sys
from types import FrameType

from fastapi import FastAPI
import uvicorn

from .config import get_settings
from .enums import NodeRole

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Placeholder for future service implementations
def create_gateway_app() -> FastAPI:
    """
    Create the FastAPI application for the Gateway service (Node 0).

    The gateway receives client requests, orchestrates the pipeline across
    retrieval and generation services, and returns final responses.

    Returns:
        FastAPI: Configured FastAPI application for gateway
    """
    from .services.gateway_app import app

    return app


def create_retrieval_app() -> FastAPI:
    """
    Create the FastAPI application for the Retrieval service (Node 1).

    The retrieval service handles:
    - Embedding generation
    - FAISS ANN search
    - Document fetching
    - Document reranking

    Returns:
        FastAPI: Configured FastAPI application for retrieval
    """
    from .services.retrieval import app

    return app


def create_generation_app() -> FastAPI:
    """
    Create the FastAPI application for the Generation service (Node 2).

    The generation service handles:
    - LLM response generation
    - Sentiment analysis
    - Safety/toxicity filtering

    Returns:
        FastAPI: Configured FastAPI application for generation
    """
    from .services.generation import app

    return app


def create_app_for_role(role: NodeRole) -> FastAPI:
    """
    Create the appropriate FastAPI app based on node role.

    Args:
        role: The role of this node (gateway/retrieval/generation)

    Returns:
        FastAPI: The configured application for this role

    Raises:
        ValueError: If role is not recognized
    """
    if role == NodeRole.GATEWAY:
        logger.info("Creating Gateway application (Node 0)")
        return create_gateway_app()
    elif role == NodeRole.RETRIEVAL:
        logger.info("Creating Retrieval application (Node 1)")
        return create_retrieval_app()
    elif role == NodeRole.GENERATION:
        logger.info("Creating Generation application (Node 2)")
        return create_generation_app()
    else:
        raise ValueError(f"Unknown role: {role}")


def setup_signal_handlers(server: uvicorn.Server) -> None:
    """
    Setup graceful shutdown handlers for SIGINT and SIGTERM.

    Args:
        server: The uvicorn server instance to shutdown
    """

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        logger.info("Received signal %d, initiating graceful shutdown...", signum)
        server.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main() -> None:
    """
    Main entry point for the pipeline runtime.
    """
    try:
        # Load configuration
        settings = get_settings()

        # Configure logging level from settings
        logging.getLogger().setLevel(settings.log_level)
        logger.setLevel(settings.log_level)

        # Log startup information
        logger.info("=" * 70)
        logger.info("ML Pipeline - Node Startup")
        logger.info("=" * 70)
        logger.info("Node Number: %d", settings.node_number)
        logger.info("Total Nodes: %d", settings.total_nodes)
        logger.info("Node Role: %s", settings.role.value)
        logger.info("Listen Address: %s:%d", settings.listen_host, settings.listen_port)
        logger.info("Gateway URL: %s", settings.gateway_url)
        logger.info("Retrieval URL: %s", settings.retrieval_url)
        logger.info("Generation URL: %s", settings.generation_url)
        logger.info("FAISS Index: %s", settings.faiss_index_path)
        logger.info("Documents Dir: %s", settings.documents_dir)
        logger.info("CPU Only: %s", settings.only_cpu)
        logger.info("=" * 70)

        # Create the appropriate app for this node's role
        app = create_app_for_role(settings.role)

        # Configure uvicorn server
        config = uvicorn.Config(
            app,
            host=settings.listen_host,
            port=settings.listen_port,
            log_level=settings.log_level.lower(),
            access_log=True,
            server_header=False,
            date_header=False,
        )

        server = uvicorn.Server(config)

        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(server)

        # Start the server
        logger.info(
            "Starting %s service on http://%s:%d",
            settings.role.value,
            settings.listen_host,
            settings.listen_port,
        )
        logger.info("Service is ready to accept requests")
        logger.info("Press Ctrl+C to shutdown")

        # Run the server
        server.run()

        logger.info("%s service shutdown complete", settings.role.value)

    except Exception as e:
        logger.exception("Fatal error during startup: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
