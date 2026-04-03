"""Testes para football_moneyball.domain.embeddings — embeddings hexagonais.

Estes testes ja eram puros (generate_embeddings, cluster_players, _find_optimal_k
nao dependem de I/O), entao a migracao e troca de import paths e rename de
_GROUP_ARCHETYPES -> GROUP_ARCHETYPES (agora em constants).
"""

import numpy as np
import pandas as pd
import pytest

from football_moneyball.domain.constants import GROUP_ARCHETYPES
from football_moneyball.domain.embeddings import (
    generate_embeddings,
    cluster_players,
    _find_optimal_k,
)


class TestGroupArchetypes:
    def test_has_all_position_groups(self):
        for group in ["DEF", "MID", "FWD", "GK"]:
            assert group in GROUP_ARCHETYPES

    def test_each_group_has_archetypes(self):
        for group, archetypes in GROUP_ARCHETYPES.items():
            assert len(archetypes) >= 3, f"{group} has too few archetypes"


class TestFindOptimalK:
    def test_finds_reasonable_k(self):
        rng = np.random.RandomState(42)
        # 3 clear clusters
        X = np.vstack([
            rng.randn(20, 5) + [0, 0, 0, 0, 0],
            rng.randn(20, 5) + [5, 5, 5, 5, 5],
            rng.randn(20, 5) + [10, 10, 10, 10, 10],
        ])
        k = _find_optimal_k(X, k_range=range(2, 8))
        assert 2 <= k <= 5  # Should find 3 or close

    def test_handles_small_dataset(self):
        X = np.random.randn(5, 3)
        k = _find_optimal_k(X, k_range=range(2, 10))
        assert k < len(X)


class TestGenerateEmbeddings:
    def _make_profiles(self) -> pd.DataFrame:
        """Cria perfis sinteticos com posicao."""
        rng = np.random.RandomState(42)
        rows = []
        for i in range(30):
            group = ["DEF", "MID", "FWD"][i % 3]
            rows.append({
                "player_id": i,
                "player_name": f"Player_{i}",
                "team": "Test",
                "position_group": group,
                "goals": rng.rand() * (3 if group == "FWD" else 0.5),
                "tackles": rng.rand() * (3 if group == "DEF" else 0.5),
                "key_passes": rng.rand() * (3 if group == "MID" else 0.5),
                "passes_completed": rng.rand() * 50,
                "minutes_played": 80 + rng.rand() * 10,
            })
        return pd.DataFrame(rows)

    def test_generates_embeddings_per_group(self):
        profiles = self._make_profiles()
        result_df, pca_dict = generate_embeddings(profiles)
        assert "embedding" in result_df.columns
        assert len(result_df) == 30
        # Should have PCA for each group
        assert isinstance(pca_dict, dict)

    def test_embedding_dimension(self):
        profiles = self._make_profiles()
        result_df, _ = generate_embeddings(profiles, n_components=8)
        # Each embedding should be a list of floats
        for emb in result_df["embedding"]:
            assert isinstance(emb, list)
            assert len(emb) <= 8


class TestClusterPlayers:
    def test_assigns_archetypes(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "player_id": range(20),
            "player_name": [f"P{i}" for i in range(20)],
            "team": "Test",
            "position_group": ["DEF"] * 10 + ["MID"] * 10,
            "embedding": [rng.randn(8).tolist() for _ in range(20)],
        })
        result = cluster_players(df)
        assert "cluster_label" in result.columns
        assert "archetype" in result.columns
        assert result["archetype"].notna().all()
