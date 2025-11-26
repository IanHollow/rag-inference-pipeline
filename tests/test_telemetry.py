"""
Tests for the telemetry module.

Tests cover:
- ResourceSnapshot creation and serialization
- StageProfiler functionality
- SampledStageProfiler with sampling
- profile_context context manager
"""

import time
from unittest.mock import MagicMock, patch

from pipeline.telemetry.profiling import (
    ProfileResult,
    SampledStageProfiler,
    StageProfiler,
    get_resource_snapshot,
    profile_context,
)


class TestResourceSnapshot:
    """Tests for ResourceSnapshot class."""

    def test_get_resource_snapshot_returns_valid_data(self) -> None:
        """Test that get_resource_snapshot returns valid data."""
        snapshot = get_resource_snapshot()

        assert isinstance(snapshot.timestamp, float)
        assert snapshot.timestamp > 0
        assert isinstance(snapshot.rss, int)
        assert snapshot.rss >= 0
        assert isinstance(snapshot.vms, int)
        assert snapshot.vms >= 0
        assert isinstance(snapshot.memory_percent, float)
        assert 0 <= snapshot.memory_percent <= 100
        assert isinstance(snapshot.cpu_percent, float)
        assert isinstance(snapshot.memory_mb, float)
        assert snapshot.memory_mb >= 0

    def test_snapshot_to_dict(self) -> None:
        """Test that snapshot converts to dictionary correctly."""
        snapshot = get_resource_snapshot()

        result = snapshot.to_dict()

        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "rss" in result
        assert "vms" in result
        assert "memory_percent" in result
        assert "cpu_percent" in result
        assert "memory_mb" in result

    def test_memory_mb_calculated_correctly(self) -> None:
        """Test that memory_mb is correctly derived from rss."""
        snapshot = get_resource_snapshot()

        expected_mb = snapshot.rss / (1024 * 1024)
        assert abs(snapshot.memory_mb - expected_mb) < 0.01


class TestProfileResult:
    """Tests for ProfileResult class."""

    def test_profile_result_to_dict(self) -> None:
        """Test that ProfileResult converts to dictionary correctly."""
        before = get_resource_snapshot()
        time.sleep(0.01)
        after = get_resource_snapshot()

        result = ProfileResult(
            stage_name="test_stage",
            duration_ms=10.5,
            before=before,
            after=after,
            memory_delta_mb=1.5,
            peak_memory_mb=100.0,
        )

        result_dict = result.to_dict()

        assert result_dict["stage_name"] == "test_stage"
        assert result_dict["duration_ms"] == 10.5
        assert result_dict["memory_delta_mb"] == 1.5
        assert result_dict["peak_memory_mb"] == 100.0
        assert "before" in result_dict
        assert "after" in result_dict


class TestProfileContext:
    """Tests for profile_context context manager."""

    def test_profile_context_measures_duration(self) -> None:
        """Test that profile_context measures duration correctly."""
        with profile_context("test_stage", log_results=False) as result:
            time.sleep(0.1)

        # Duration should be approximately 100ms
        assert result.duration_ms >= 90  # Allow some tolerance
        assert result.duration_ms < 200

    def test_profile_context_captures_snapshots(self) -> None:
        """Test that profile_context captures before/after snapshots."""
        with profile_context("test_stage", log_results=False) as result:
            pass

        assert result.before is not None
        assert result.after is not None
        assert result.after.timestamp >= result.before.timestamp

    def test_profile_context_stage_name(self) -> None:
        """Test that profile_context sets stage name correctly."""
        with profile_context("my_custom_stage", log_results=False) as result:
            pass

        assert result.stage_name == "my_custom_stage"

    def test_profile_context_calculates_memory_delta(self) -> None:
        """Test that memory delta is calculated."""
        with profile_context("test_stage", log_results=False) as result:
            # Allocate some memory
            _ = [0] * 100000

        # Memory delta should be calculated (may be positive or negative due to GC)
        assert isinstance(result.memory_delta_mb, float)

    @patch("pipeline.telemetry.profiling.logger")
    def test_profile_context_logs_when_enabled(self, mock_logger: MagicMock) -> None:
        """Test that profile_context logs results when log_results=True."""
        with profile_context("logged_stage", log_results=True):
            pass

        mock_logger.info.assert_called()

    @patch("pipeline.telemetry.profiling.logger")
    def test_profile_context_no_log_when_disabled(self, mock_logger: MagicMock) -> None:
        """Test that profile_context doesn't log when log_results=False."""
        with profile_context("silent_stage", log_results=False):
            pass

        mock_logger.info.assert_not_called()


class TestStageProfiler:
    """Tests for StageProfiler class."""

    def test_profiler_add_profile(self) -> None:
        """Test adding profiles to profiler."""
        profiler = StageProfiler()
        before = get_resource_snapshot()
        after = get_resource_snapshot()

        profile = ProfileResult(
            stage_name="test",
            duration_ms=10.0,
            before=before,
            after=after,
            memory_delta_mb=0.0,
            peak_memory_mb=100.0,
        )

        profiler.add(profile)

        assert len(profiler.profiles) == 1

    def test_profiler_get_summary_empty(self) -> None:
        """Test summary with no profiles."""
        profiler = StageProfiler()

        summary = profiler.get_summary()

        assert summary["total_duration_ms"] == 0.0
        assert summary["stages"] == []

    def test_profiler_get_summary_multiple_profiles(self) -> None:
        """Test summary with multiple profiles."""
        profiler = StageProfiler()
        before = get_resource_snapshot()
        after = get_resource_snapshot()

        for i in range(3):
            profile = ProfileResult(
                stage_name=f"stage_{i}",
                duration_ms=10.0 * (i + 1),
                before=before,
                after=after,
                memory_delta_mb=1.0,
                peak_memory_mb=100.0 + i,
            )
            profiler.add(profile)

        summary = profiler.get_summary()

        assert summary["total_duration_ms"] == 60.0  # 10 + 20 + 30
        assert summary["stage_count"] == 3
        assert summary["peak_memory_mb"] == 102.0
        assert summary["total_memory_delta_mb"] == 3.0
        assert len(summary["stages"]) == 3


class TestSampledStageProfiler:
    """Tests for SampledStageProfiler class."""

    def test_profiler_disabled(self) -> None:
        """Test that profiler is disabled when enabled=False."""
        profiler = SampledStageProfiler(enabled=False, sample_rate=1.0)

        assert not profiler.is_active()
        assert profiler.profiler is None

    def test_profiler_disabled_with_zero_sample_rate(self) -> None:
        """Test that profiler is disabled when sample_rate=0."""
        profiler = SampledStageProfiler(enabled=True, sample_rate=0.0)

        assert not profiler.enabled

    @patch("secrets.SystemRandom")
    def test_profiler_enabled_and_sampled(self, mock_random: MagicMock) -> None:
        """Test that profiler is active when sampled."""
        mock_random.return_value.random.return_value = 0.5  # Below sample_rate

        profiler = SampledStageProfiler(enabled=True, sample_rate=1.0)

        assert profiler.is_active()
        assert profiler.profiler is not None

    @patch("secrets.SystemRandom")
    def test_profiler_not_sampled(self, mock_random: MagicMock) -> None:
        """Test that profiler is not active when not sampled."""
        mock_random.return_value.random.return_value = 0.9  # Above sample_rate

        profiler = SampledStageProfiler(enabled=True, sample_rate=0.5)

        assert not profiler.is_active()

    @patch("secrets.SystemRandom")
    def test_track_when_active(self, mock_random: MagicMock) -> None:
        """Test track method when profiler is active."""
        mock_random.return_value.random.return_value = 0.1

        profiler = SampledStageProfiler(enabled=True, sample_rate=1.0)

        with profiler.track("test_stage") as result:
            time.sleep(0.01)

        assert result is not None
        assert result.stage_name == "test_stage"
        assert profiler.profiler is not None
        assert len(profiler.profiler.profiles) == 1

    def test_track_when_inactive(self) -> None:
        """Test track method when profiler is inactive."""
        profiler = SampledStageProfiler(enabled=False, sample_rate=1.0)

        with profiler.track("test_stage") as result:
            pass

        assert result is None

    @patch("secrets.SystemRandom")
    def test_summary_when_active(self, mock_random: MagicMock) -> None:
        """Test summary method when profiler is active."""
        mock_random.return_value.random.return_value = 0.1

        profiler = SampledStageProfiler(enabled=True, sample_rate=1.0)

        with profiler.track("stage1"):
            pass
        with profiler.track("stage2"):
            pass

        summary = profiler.summary()

        assert summary is not None
        assert summary["stage_count"] == 2

    def test_summary_when_inactive(self) -> None:
        """Test summary method when profiler is inactive."""
        profiler = SampledStageProfiler(enabled=False, sample_rate=1.0)

        summary = profiler.summary()

        assert summary is None

    @patch("secrets.SystemRandom")
    def test_clear_profiles(self, mock_random: MagicMock) -> None:
        """Test clearing profiler data."""
        mock_random.return_value.random.return_value = 0.1

        profiler = SampledStageProfiler(enabled=True, sample_rate=1.0)

        with profiler.track("stage1"):
            pass

        assert profiler.profiler is not None
        assert len(profiler.profiler.profiles) == 1

        profiler.clear()

        assert len(profiler.profiler.profiles) == 0

    def test_sample_rate_clamped(self) -> None:
        """Test that sample_rate is clamped to [0, 1]."""
        profiler_high = SampledStageProfiler(enabled=True, sample_rate=2.0)
        profiler_low = SampledStageProfiler(enabled=True, sample_rate=-0.5)

        assert profiler_high.sample_rate == 1.0
        assert profiler_low.sample_rate == 0.0

    def test_custom_logger(self) -> None:
        """Test that custom logger is used."""
        custom_logger = MagicMock()

        profiler = SampledStageProfiler(enabled=False, sample_rate=1.0, logger=custom_logger)

        assert profiler.logger == custom_logger
