"""
Tests for the OpenTelemetry tracing module.

Tests cover:
- setup_tracing initialization with various settings
- Thread-safe singleton pattern
- Resource attributes configuration
- Exporter configuration
- instrument_fastapi_app function
"""

from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
import pytest

from pipeline.config import PipelineSettings


class TestSetupTracing:
    """Tests for the setup_tracing function."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings with tracing enabled."""
        return PipelineSettings(
            NODE_NUMBER=1,
            ENABLE_TRACING=True,
            OTEL_EXPORTER_ENDPOINT="http://localhost:4317",
            OTEL_EXPORTER_INSECURE=True,
        )

    @pytest.fixture
    def settings_tracing_disabled(self) -> PipelineSettings:
        """Create test settings with tracing disabled."""
        return PipelineSettings(
            NODE_NUMBER=1,
            ENABLE_TRACING=False,
        )

    def test_setup_tracing_disabled(self, settings_tracing_disabled: PipelineSettings) -> None:
        """Test that setup_tracing does nothing when tracing is disabled."""
        from pipeline.telemetry import tracing

        # Reset state
        tracing._configured = False

        with patch.object(tracing, "TracerProvider") as mock_provider:
            tracing.setup_tracing(settings_tracing_disabled, "test-service")

            mock_provider.assert_not_called()
            assert not tracing._configured

    @patch("pipeline.telemetry.tracing.HTTPXClientInstrumentor")
    @patch("pipeline.telemetry.tracing.OTLPSpanExporter")
    @patch("pipeline.telemetry.tracing.trace")
    def test_setup_tracing_configures_provider(
        self,
        mock_trace: MagicMock,
        mock_exporter: MagicMock,
        mock_httpx_instr: MagicMock,
        settings: PipelineSettings,
    ) -> None:
        """Test that setup_tracing configures the tracer provider."""
        from pipeline.telemetry import tracing

        # Reset state
        tracing._configured = False

        mock_exporter.return_value = MagicMock()
        mock_httpx_instr.return_value.instrument = MagicMock()

        tracing.setup_tracing(settings, "test-service")

        # Verify trace provider was set
        mock_trace.set_tracer_provider.assert_called_once()
        provider = mock_trace.set_tracer_provider.call_args[0][0]
        assert isinstance(provider, TracerProvider)

        # Verify OTLP exporter was created
        mock_exporter.assert_called_once_with(
            endpoint=settings.otel_exporter_endpoint,
            insecure=settings.otel_exporter_insecure,
            timeout=5,
        )

        # Verify configured flag is set
        assert tracing._configured

        # Reset for other tests
        tracing._configured = False

    @patch("pipeline.telemetry.tracing.HTTPXClientInstrumentor")
    @patch("pipeline.telemetry.tracing.OTLPSpanExporter")
    @patch("pipeline.telemetry.tracing.trace")
    def test_setup_tracing_idempotent(
        self,
        mock_trace: MagicMock,
        mock_exporter: MagicMock,
        mock_httpx_instr: MagicMock,
        settings: PipelineSettings,
    ) -> None:
        """Test that setup_tracing is idempotent (only runs once)."""
        from pipeline.telemetry import tracing

        # Reset state
        tracing._configured = False

        mock_exporter.return_value = MagicMock()
        mock_httpx_instr.return_value.instrument = MagicMock()

        # Call twice
        tracing.setup_tracing(settings, "test-service")
        tracing.setup_tracing(settings, "test-service")

        # Should only configure once
        mock_trace.set_tracer_provider.assert_called_once()

        # Reset for other tests
        tracing._configured = False

    @patch("pipeline.telemetry.tracing.HTTPXClientInstrumentor")
    @patch("pipeline.telemetry.tracing.ConsoleSpanExporter")
    @patch("pipeline.telemetry.tracing.OTLPSpanExporter")
    @patch("pipeline.telemetry.tracing.trace")
    def test_setup_tracing_fallback_to_console(
        self,
        mock_trace: MagicMock,
        mock_otlp_exporter: MagicMock,
        mock_console_exporter: MagicMock,
        mock_httpx_instr: MagicMock,
        settings: PipelineSettings,
    ) -> None:
        """Test that setup_tracing falls back to console exporter on OTLP failure."""
        from pipeline.telemetry import tracing

        # Reset state
        tracing._configured = False

        # Make OTLP fail
        mock_otlp_exporter.side_effect = Exception("Connection failed")
        mock_console_exporter.return_value = MagicMock()
        mock_httpx_instr.return_value.instrument = MagicMock()

        tracing.setup_tracing(settings, "test-service")

        # Console exporter should be used
        mock_console_exporter.assert_called_once()

        # Reset for other tests
        tracing._configured = False


class TestInstrumentFastAPIApp:
    """Tests for the instrument_fastapi_app function."""

    @patch("pipeline.telemetry.tracing.FastAPIInstrumentor")
    def test_instrument_app(self, mock_instrumentor: MagicMock) -> None:
        """Test that FastAPI app is instrumented."""
        from pipeline.telemetry.tracing import instrument_fastapi_app

        mock_app = MagicMock()

        instrument_fastapi_app(mock_app)

        mock_instrumentor.instrument_app.assert_called_once_with(
            mock_app,
            excluded_urls="/metrics,/health",
        )

    @patch("pipeline.telemetry.tracing.FastAPIInstrumentor")
    def test_instrument_app_handles_error(self, mock_instrumentor: MagicMock) -> None:
        """Test that errors during instrumentation are handled gracefully."""
        from pipeline.telemetry.tracing import instrument_fastapi_app

        mock_instrumentor.instrument_app.side_effect = Exception("Instrumentation failed")
        mock_app = MagicMock()

        # Should not raise
        instrument_fastapi_app(mock_app)

        mock_instrumentor.instrument_app.assert_called_once()


class TestResourceConfiguration:
    """Tests for resource attribute configuration."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings. Node 1 is retrieval in a 3-node cluster."""
        return PipelineSettings(
            NODE_NUMBER=1,
            ENABLE_TRACING=True,
            OTEL_EXPORTER_ENDPOINT="http://localhost:4317",
        )

    @patch("pipeline.telemetry.tracing.HTTPXClientInstrumentor")
    @patch("pipeline.telemetry.tracing.OTLPSpanExporter")
    @patch("pipeline.telemetry.tracing.trace")
    @patch("pipeline.telemetry.tracing.Resource")
    def test_resource_attributes(
        self,
        mock_resource: MagicMock,
        mock_trace: MagicMock,
        mock_exporter: MagicMock,
        mock_httpx_instr: MagicMock,
        settings: PipelineSettings,
    ) -> None:
        """Test that resource attributes are properly configured."""
        from pipeline.telemetry import tracing

        # Reset state
        tracing._configured = False

        mock_exporter.return_value = MagicMock()
        mock_httpx_instr.return_value.instrument = MagicMock()
        mock_resource.create.return_value = MagicMock()

        tracing.setup_tracing(settings, "test-retrieval-service")

        # Verify Resource.create was called with expected attributes
        mock_resource.create.assert_called_once()
        call_args = mock_resource.create.call_args[0][0]

        assert call_args["service.name"] == "test-retrieval-service"
        assert call_args["service.namespace"] == "cs5416-ml-pipeline"
        assert call_args["service.instance.id"] == "node-1"
        assert call_args["pipeline.node"] == 1
        assert call_args["pipeline.role"] == "retrieval"

        # Reset for other tests
        tracing._configured = False
