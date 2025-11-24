"""
Runtime entry point for the distributed ML pipeline.

This module is invoked via `python -m pipeline.runtime`
"""

import logging
import os
import signal
import sys
from types import FrameType

from fastapi import FastAPI
import uvicorn

from .config import PipelineSettings, get_settings
from .runtime_factory import create_app_from_profile
from .telemetry import setup_tracing
from .utils.executors import ServiceExecutorFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def configure_libraries(settings: PipelineSettings) -> None:
    """Configure library thread counts based on settings."""
    # Set OMP_NUM_THREADS for libraries that respect it
    os.environ["OMP_NUM_THREADS"] = str(settings.cpu_inference_threads)

    try:
        import torch

        torch.set_num_threads(settings.cpu_inference_threads)
        torch.set_num_interop_threads(max(2, settings.cpu_inference_threads // 2))
        logger.info(
            "Configured torch threads: intra=%d, inter=%d",
            settings.cpu_inference_threads,
            max(2, settings.cpu_inference_threads // 2),
        )
    except ImportError:
        pass

    try:
        import faiss

        faiss.omp_set_num_threads(settings.cpu_inference_threads)
        logger.info("Configured faiss threads: %d", settings.cpu_inference_threads)
    except ImportError:
        pass


def create_app() -> FastAPI:
    """
    Create the FastAPI application based on configuration.

    Returns:
        FastAPI: The configured application
    """
    settings = get_settings()
    configure_libraries(settings)

    # Initialize executor factory with settings
    ServiceExecutorFactory.initialize(settings)

    logger.info("Creating application for node %d", settings.node_number)
    return create_app_from_profile(settings)


def setup_signal_handlers(server: uvicorn.Server) -> None:
    """
    Setup graceful shutdown handlers for SIGINT and SIGTERM.

    Args:
        server: The uvicorn server instance to shutdown
    """

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        logger.info("Received signal %d, initiating graceful shutdown...", signum)
        ServiceExecutorFactory.shutdown()
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

        # Initialize tracing before apps/modules are imported
        setup_tracing(settings, service_name=f"pipeline-{settings.role.value}")

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
        app = create_app()

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
