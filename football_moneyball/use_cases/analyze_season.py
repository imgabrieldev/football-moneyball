"""Use case: analise of a season complete of a time."""

from __future__ import annotations

from typing import Any

import pandas as pd


class AnalyzeSeason:
    """Process todas as matches of a time in a season.

    Parameters
    ----------
    provider : DataProvider
        Fonte of data.
    repo : MatchRepository
        Repositorio for persistencia.
    """

    def __init__(self, provider, repo) -> None:
        self.provider = provider
        self.repo = repo

    def execute(
        self,
        competition: str,
        season: str,
        team: str,
        competition_id: int,
        season_id: int,
        refresh: bool = False,
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """Process the season inteira.

        Parameters
        ----------
        competition, season, team : str
            Filtros of competicao/season/time.
        competition_id, season_id : int
            IDs for busca of matches in the provider.
        refresh : bool
            Forcar reprocessamento.
        on_progress : callable, optional
            Callback(match_index, total, match_info) for progresso.

        Returns
        -------
        dict
            Chaves: 'combined_df', 'agg_stats', 'team_matches_count',
            'total_players'.
        """
        from football_moneyball.domain import metrics, pressing, network

        matches = self.provider.get_matches(competition_id, season_id)
        team_matches = matches[
            (matches.get("home_team", matches.get("home_team_name", "")) == team)
            | (matches.get("away_team", matches.get("away_team_name", "")) == team)
        ]

        if team_matches.empty:
            return {"error": f"Nenhuma match encontrada for '{team}'."}

        all_metrics = []
        total = len(team_matches)

        for i, (_, match_row) in enumerate(team_matches.iterrows()):
            mid = int(match_row["match_id"])

            if on_progress:
                on_progress(i, total, match_row)

            if not refresh and self.repo.match_exists(mid):
                match_metrics = self.repo.get_match_data(mid)
            else:
                try:
                    events = self.provider.get_match_events(mid)
                    if events.empty:
                        continue
                    match_metrics = metrics.extract_match_metrics(events)
                    if match_metrics.empty:
                        continue

                    # Persist
                    match_info = {
                        "match_id": mid,
                        "competition": competition,
                        "season": season,
                        "match_date": str(match_row.get("match_date", "")),
                        "home_team": match_row.get("home_team", ""),
                        "away_team": match_row.get("away_team", ""),
                        "home_score": int(match_row.get("home_score", 0)),
                        "away_score": int(match_row.get("away_score", 0)),
                    }
                    self.repo.save_match(match_info)
                    self.repo.save_player_metrics(match_metrics, mid)

                    # Pass network
                    try:
                        graph, edges_df = network.build_pass_network(events, team=team)
                        edge_features = network.compute_edge_features(graph)
                        feat_list = []
                        for _, erow in edges_df.iterrows():
                            key = (erow["passer_id"], erow["receiver_id"])
                            feat_list.append(edge_features.get(key, {}))
                        edges_df["features"] = feat_list
                        self.repo.save_pass_network(edges_df, mid)
                    except Exception:
                        pass

                    # Pressing
                    try:
                        pressing_df = pressing.compute_match_pressing(events)
                        if not pressing_df.empty:
                            self.repo.save_pressing_metrics(pressing_df, mid)
                    except Exception:
                        pass

                except Exception:
                    continue

            if not match_metrics.empty:
                all_metrics.append(match_metrics)

        if not all_metrics:
            return {"error": "Nenhuma metric extraida."}

        combined = pd.concat(all_metrics, ignore_index=True)
        team_data = combined[combined["team"] == team]

        if team_data.empty:
            return {"error": "Nenhuma metric for the time especificado."}

        # Aggregate
        numeric_cols = team_data.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("player_id", "match_id")]

        agg_stats = (
            team_data.groupby(["player_id", "player_name"])[numeric_cols]
            .sum()
            .reset_index()
        )
        agg_stats["matches"] = team_data.groupby("player_id").size().values
        agg_stats = agg_stats.sort_values("xg", ascending=False)

        return {
            "combined_df": combined,
            "agg_stats": agg_stats,
            "team_matches_count": total,
            "total_players": len(agg_stats),
            "team": team,
            "competition": competition,
            "season": season,
        }

    def generate_embeddings(self, competition: str, season: str) -> bool:
        """Generate embeddings for todos os players of the season.

        Returns
        -------
        bool
            True if gerou with sucesso.
        """
        from football_moneyball.domain import embeddings

        all_metrics = self.repo.get_all_metrics(competition, season)
        if all_metrics.empty:
            return False

        profiles = embeddings.build_player_profiles(all_metrics)
        if profiles.empty:
            return False

        emb_df, pca = embeddings.generate_embeddings(profiles)
        emb_df = embeddings.cluster_players(emb_df, pca=pca)
        emb_df["season"] = season
        emb_df["competition"] = competition
        self.repo.save_embeddings(emb_df)
        return True

    def compute_rapm(self, competition: str, season: str) -> bool:
        """Compute RAPM for the season.

        Returns
        -------
        bool
            True if calculou with sucesso.
        """
        try:
            from football_moneyball.domain import rapm

            season_matches = self.repo.get_season_matches(competition, season)
            if not season_matches:
                return False

            all_stints = []
            for match in season_matches:
                cached = self.repo.get_cached_stints(match.match_id)
                if not cached.empty:
                    all_stints.append(cached)
                else:
                    events = self.provider.get_match_events(match.match_id)
                    lineups = self.provider.get_lineups(match.match_id)
                    stints_df = rapm.reconstruct_stints(events, lineups)
                    if not stints_df.empty:
                        self.repo.save_stints(stints_df, match.match_id)
                        all_stints.append(stints_df)

            if not all_stints:
                return False

            combined = pd.concat(all_stints, ignore_index=True)
            X, y, player_ids = rapm.build_rapm_matrix(combined)
            alpha = rapm.cross_validate_alpha(X, y)
            results = rapm.fit_rapm(X, y, player_ids, alpha=alpha)
            return not results.empty
        except Exception:
            return False
