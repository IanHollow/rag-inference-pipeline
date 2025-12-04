import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pytest

from pipeline.config import PipelineSettings
from pipeline.config.profile_schema import ComponentConfig, ProfileFile, RouteConfig
from pipeline.enums import ComponentType
from pipeline.runtime_factory import create_app_from_profile, load_role_profile


class TestRuntimeFactory(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = PipelineSettings(
            node_number=0, pipeline_role_profile="custom", role_profile_override_path=None
        )
        # Reset global executors to ensure clean state
        from pipeline.services.generation import api as generation_api
        from pipeline.services.retrieval import api as retrieval_api

        retrieval_api._executor_container.instance = None
        generation_api._executor_container.instance = None

    @patch("pipeline.runtime_factory.create_component")
    @patch("pipeline.runtime_factory.load_role_profile")
    def test_create_app_custom_profile(
        self, mock_load_profile: MagicMock, mock_create_component: MagicMock
    ) -> None:
        # Setup mock profile
        profile = ProfileFile(
            name="test_profile",
            components=[
                ComponentConfig(name="my_embedding", type="embedding_generator"),
                ComponentConfig(name="my_faiss", type="faiss_store"),
            ],
            routes=[
                RouteConfig(
                    target="retrieval",
                    prefix="/custom_retrieve",
                    component_aliases={"embedding_generator": "my_embedding"},
                )
            ],
        )
        mock_load_profile.return_value = profile

        # Mock component creation
        mock_component = MagicMock()
        mock_create_component.return_value = mock_component

        # Create app
        app = create_app_from_profile(self.settings)

        # Verify routes
        routes = [getattr(r, "path", "") for r in app.routes]
        # The retrieval router has / endpoint, so mounted at /custom_retrieve it becomes /custom_retrieve
        assert "/custom_retrieve" in routes

        # Verify aliases
        assert app.state.component_aliases["embedding_generator"] == "my_embedding"

        # Verify registry
        assert "my_embedding" in app.state.registry.components

    @patch("pipeline.runtime_factory.yaml.safe_load")
    @patch(
        "pathlib.Path.open",
        new_callable=unittest.mock.mock_open,
        read_data="name: test\ncomponents: []\nroutes: []",
    )
    @patch("pipeline.runtime_factory.Path.exists")
    def test_load_role_profile_from_yaml(
        self, mock_exists: MagicMock, _mock_open: MagicMock, mock_yaml: MagicMock
    ) -> None:
        mock_exists.return_value = True
        mock_yaml.return_value = {"name": "test_profile", "components": [], "routes": []}

        self.settings.pipeline_role_profile = "test_profile"
        profile = load_role_profile(self.settings)

        assert profile.name == "test_profile"

    @patch("pipeline.runtime_factory.create_component")
    @patch("pipeline.runtime_factory.load_role_profile")
    def test_lifecycle_events(
        self, mock_load_profile: MagicMock, mock_create_component: MagicMock
    ) -> None:
        profile = ProfileFile(
            name="lifecycle_test",
            components=[ComponentConfig(name="comp1", type="embedding_generator")],
            routes=[],
        )
        mock_load_profile.return_value = profile

        mock_comp = MagicMock()
        mock_comp.start = AsyncMock()
        mock_comp.stop = AsyncMock()
        mock_create_component.return_value = mock_comp

        app = create_app_from_profile(self.settings)

        # Trigger startup/shutdown via TestClient
        with TestClient(app):
            # Startup happens on enter
            pass
        # Shutdown happens on exit

        mock_comp.start.assert_called()
        mock_comp.stop.assert_called()

    def test_profile_validation_aliases(self) -> None:
        # Test that invalid alias raises ValueError
        with pytest.raises(ValueError, match="points to unknown component") as cm:
            ProfileFile(
                name="invalid_alias",
                components=[ComponentConfig(name="comp1", type="type1")],
                routes=[
                    RouteConfig(target="gateway", component_aliases={"dep1": "non_existent_comp"})
                ],
            )
        assert "points to unknown component" in str(cm.value)

    @patch("pipeline.runtime_factory.create_component")
    @patch("pipeline.runtime_factory.load_role_profile")
    def test_integration_renamed_components(
        self, mock_load_profile: MagicMock, mock_create_component: MagicMock
    ) -> None:
        # Test that a profile with renamed components and aliases works end-to-end
        profile = ProfileFile(
            name="integration_test",
            components=[
                ComponentConfig(
                    name="custom_embedding",
                    type="embedding_generator",
                    aliases=["embedding_generator"],  # Explicit alias on component
                ),
                ComponentConfig(name="custom_faiss", type="faiss_store", aliases=["faiss_store"]),
                ComponentConfig(
                    name="custom_docs", type="document_store", aliases=["document_store"]
                ),
            ],
            routes=[RouteConfig(target="retrieval", prefix="/retrieve")],
        )
        mock_load_profile.return_value = profile

        # Mock components
        mock_emb = MagicMock()
        mock_emb.is_loaded = True
        mock_emb.encode.return_value = np.array([[0.1]], dtype=np.float32)  # Mock embedding return

        mock_faiss = MagicMock()
        mock_faiss.is_loaded = True
        # Mock search return
        mock_indices = MagicMock()
        mock_indices.tolist.return_value = [1]
        mock_dists = MagicMock()
        mock_dists.tolist.return_value = [0.1]
        mock_faiss.search.return_value = ([mock_dists], [mock_indices])

        mock_docs = MagicMock()
        mock_docs.fetch_documents_batch.return_value = [
            [MagicMock(doc_id=1, title="t", content="c", category="cat")]
        ]

        # Configure create_component to return appropriate mocks
        def side_effect(ctype: object, _settings: object, _config: object) -> MagicMock:
            if ctype in ("embedding_generator", ComponentType.EMBEDDING):
                return mock_emb
            if ctype in ("faiss_store", ComponentType.FAISS):
                return mock_faiss
            if ctype in ("document_store", ComponentType.DOCUMENT_STORE):
                return mock_docs
            return MagicMock()

        mock_create_component.side_effect = side_effect

        app = create_app_from_profile(self.settings)

        with TestClient(app) as client:
            # Hit the retrieval endpoint
            # We need to mock the request body
            response = client.post(
                "/retrieve",
                json={"batch_id": "test_batch", "items": [{"request_id": "1", "query": "test"}]},
            )

            # Should not be 500 (Internal Server Error) due to missing dependency
            # It might be 422 (Validation Error) or 200 depending on mocks,
            # but definitely not 500 "Component registry not initialized" or similar
            assert response.status_code != 500

            # Verify aliases were registered
            assert app.state.component_aliases["embedding_generator"] == "custom_embedding"
            assert app.state.component_aliases["faiss_store"] == "custom_faiss"

    @patch("pipeline.runtime_factory.create_component")
    @patch("pipeline.runtime_factory.load_role_profile")
    def test_alias_resolution_integration(
        self, mock_load_profile: MagicMock, mock_create_component: MagicMock
    ) -> None:
        """
        Test that a renamed component is correctly resolved via aliases
        during a request.
        """
        # 1. Define profile with renamed component and alias
        profile = ProfileFile(
            name="alias_test",
            components=[ComponentConfig(name="custom_embedder", type="embedding_generator")],
            routes=[
                RouteConfig(
                    target="retrieval",
                    prefix="/retrieve",
                    component_aliases={"embedding_generator": "custom_embedder"},
                )
            ],
        )
        mock_load_profile.return_value = profile

        # 2. Mock the component
        mock_embedder = MagicMock()
        mock_embedder.is_loaded = True
        # Mock encode method which is called by retrieval endpoint
        mock_embedder.encode.return_value = np.array([[0.1, 0.2]], dtype=np.float32)

        # We need to handle multiple component creations if heuristics kick in,
        # but here we only have one component in the profile.
        # However, the factory might try to create others if heuristics are on?
        # No, factory only creates what's in components list.
        mock_create_component.return_value = mock_embedder

        # 3. Create app
        app = create_app_from_profile(self.settings)

        # 4. Verify alias registration
        assert app.state.component_aliases["embedding_generator"] == "custom_embedder"
        assert "custom_embedder" in app.state.registry.components

        # 5. Simulate request to verify dependency injection
        # We need to mock the other dependencies of retrieval service too,
        # or the endpoint will fail.
        # The retrieval endpoint needs: embedding_generator, faiss_store, document_store.
        # Our profile only has embedding_generator.
        # This means get_component("faiss_store") will return None (or fail if not in registry).

        # To make this an integration test that actually calls the endpoint,
        # we need to provide all dependencies.

        # Let's update the profile to include mocks for others
        profile.components.extend(
            [
                ComponentConfig(name="mock_faiss", type="faiss_store"),
                ComponentConfig(name="mock_docs", type="document_store"),
            ]
        )

        # Update aliases
        profile.routes[0].component_aliases.update(
            {"faiss_store": "mock_faiss", "document_store": "mock_docs"}
        )

        # Update mock_create_component to return appropriate mocks based on name/type
        # But create_component is called with ComponentConfig.
        # We can use side_effect.

        mock_faiss = MagicMock()
        mock_faiss.is_loaded = True
        # Mock return values that behave like numpy arrays
        mock_indices_row = MagicMock()
        mock_indices_row.tolist.return_value = [1]

        mock_distances_row = MagicMock()
        mock_distances_row.tolist.return_value = [0.1]

        mock_faiss.search.return_value = (
            [mock_distances_row],  # distances
            [mock_indices_row],  # indices
        )

        mock_docs = MagicMock()
        mock_docs.fetch_documents_batch.return_value = [
            [MagicMock(doc_id=1, title="t", content="c", category="cat")]
        ]

        def create_side_effect(ctype: object, _settings: object, _config: object) -> MagicMock:
            mapping = {
                "embedding_generator": mock_embedder,
                "faiss_store": mock_faiss,
                "document_store": mock_docs,
            }
            return mapping.get(str(ctype), MagicMock())

        mock_create_component.side_effect = create_side_effect

        # Re-create app with full dependencies
        app = create_app_from_profile(self.settings)

        with TestClient(app) as client:
            response = client.post(
                "/retrieve",
                json={"batch_id": "test_batch", "items": [{"request_id": "r1", "query": "test"}]},
            )

            # Check success
            assert response.status_code == 200

            # Verify our custom named component was used
            mock_embedder.encode.assert_called_once()

    @patch("pipeline.runtime_factory.create_component")
    def test_yaml_end_to_end(self, mock_create_component: MagicMock) -> None:
        """
        Test loading a profile from a real YAML file and verifying alias resolution.
        """
        from pathlib import Path
        import tempfile

        import yaml

        # Create a temporary YAML profile
        profile_data = {
            "name": "yaml_test_profile",
            "components": [
                {"name": "yaml_embedder", "type": "embedding_generator"},
                {"name": "yaml_faiss", "type": "faiss_store"},
                {"name": "yaml_docs", "type": "document_store"},
            ],
            "routes": [
                {
                    "target": "retrieval",
                    "prefix": "/yaml_retrieve",
                    "component_aliases": {
                        "embedding_generator": "yaml_embedder",
                        "faiss_store": "yaml_faiss",
                        "document_store": "yaml_docs",
                    },
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(profile_data, tmp)
            tmp_path = tmp.name

        try:
            # Configure settings to use this profile
            self.settings.role_profile_override_path = tmp_path

            # Mock components
            mock_embedder = MagicMock()
            mock_embedder.is_loaded = True
            mock_embedder.encode.return_value = np.array([[0.1]], dtype=np.float32)

            mock_faiss = MagicMock()
            mock_faiss.is_loaded = True
            mock_indices = MagicMock()
            mock_indices.tolist.return_value = [1]
            mock_dists = MagicMock()
            mock_dists.tolist.return_value = [0.1]
            mock_faiss.search.return_value = ([mock_dists], [mock_indices])

            mock_docs = MagicMock()
            mock_docs.fetch_documents_batch.return_value = [
                [MagicMock(doc_id=1, title="t", content="c", category="cat")]
            ]

            def create_side_effect(ctype: object, _settings: object, _config: object) -> MagicMock:
                # ctype can be string or Enum. Convert to string for comparison.
                type_str = ctype.value if hasattr(ctype, "value") else str(ctype)

                mapping = {
                    "embedding_generator": mock_embedder,
                    "faiss_store": mock_faiss,
                    "document_store": mock_docs,
                }
                return mapping.get(type_str, MagicMock())

            mock_create_component.side_effect = create_side_effect

            # Create app (this will call real load_role_profile reading our temp file)
            app = create_app_from_profile(self.settings)

            # Verify aliases in state
            assert app.state.component_aliases["embedding_generator"] == "yaml_embedder"

            # Hit the endpoint
            with TestClient(app) as client:
                response = client.post(
                    "/yaml_retrieve",
                    json={
                        "batch_id": "yaml_batch",
                        "items": [{"request_id": "r1", "query": "test"}],
                    },
                )

                assert response.status_code == 200
                mock_embedder.encode.assert_called()

        finally:
            p = Path(tmp_path)
            if p.exists():
                p.unlink()
