"""CLI Typer para Football Moneyball Analytics.

Interface de linha de comando com saida Rich para analise de futebol.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich import print as rprint

from football_moneyball import db, player_metrics, network_analysis, pressing

# ---------------------------------------------------------------------------
# App & globals
# ---------------------------------------------------------------------------

app = typer.Typer(name="moneyball", help="Football Moneyball Analytics CLI")

console = Console()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://moneyball:moneyball@localhost:5432/moneyball",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session():
    """Retorna uma sessao do banco de dados e exibe status de conexao."""
    try:
        session = db.get_session()
        return session
    except Exception as exc:
        console.print(f"[red]Erro ao conectar ao banco de dados: {exc}[/red]")
        raise typer.Exit(1)


def _xg_contribution(row) -> float:
    """Calcula a contribuicao xG (xG + xA) de uma linha do DataFrame."""
    xg = row.get("xg", 0) or 0
    xa = row.get("xa", 0) or 0
    return float(xg) + float(xa)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("analyze-match")
def analyze_match(
    match_id: int = typer.Argument(..., help="ID da partida StatsBomb"),
    refresh: bool = typer.Option(False, "--refresh", help="Forca reprocessamento mesmo que ja exista no banco"),
) -> None:
    """Analisa uma partida especifica e exibe metricas dos jogadores."""
    session = _get_session()

    try:
        if not refresh and db.match_exists(session, match_id):
            console.print(f"[cyan]Carregando partida {match_id} do banco de dados...[/cyan]")
            metrics_rows = (
                session.query(db.PlayerMatchMetrics)
                .filter_by(match_id=match_id)
                .all()
            )
            columns = [c.key for c in db.PlayerMatchMetrics.__table__.columns]
            metrics_data = [{col: getattr(r, col) for col in columns} for r in metrics_rows]

            import pandas as pd
            metrics_df = pd.DataFrame(metrics_data)
        else:
            with console.status("[bold green]Extraindo metricas da partida..."):
                metrics_df = player_metrics.extract_match_metrics(match_id)

            if metrics_df.empty:
                console.print(f"[red]Erro: nenhum dado encontrado para a partida {match_id}.[/red]")
                raise typer.Exit(1)

            with console.status("[bold green]Construindo rede de passes..."):
                graph, edges_df = network_analysis.build_pass_network(match_id)
                edge_features = network_analysis.compute_edge_features(graph)
                for idx, row in edges_df.iterrows():
                    key = (row["passer_id"], row["receiver_id"])
                    if key in edge_features:
                        edges_df.at[idx, "features"] = edge_features[key]

            with console.status("[bold green]Salvando no banco de dados..."):
                match_info = player_metrics.extract_match_info(match_id)
                db.upsert_match(session, match_info)
                db.upsert_player_metrics(session, metrics_df, match_id)
                db.upsert_pass_network(session, edges_df, match_id)

            # v0.2.0 — Pressing metrics
            try:
                with console.status("[bold green]Calculando metricas de pressing..."):
                    pressing_df = pressing.compute_match_pressing(match_id)
                    if not pressing_df.empty:
                        db.upsert_pressing_metrics(session, pressing_df, match_id)
            except Exception as exc:
                console.print(f"[yellow]Aviso: falha ao calcular pressing: {exc}[/yellow]")

            console.print("[green]Dados persistidos com sucesso.[/green]")

        # Display player metrics table sorted by xG contribution
        metrics_df = metrics_df.copy()
        metrics_df["xg_contribution"] = metrics_df.apply(_xg_contribution, axis=1)
        metrics_df = metrics_df.sort_values("xg_contribution", ascending=False)

        table = Table(title=f"Metricas dos Jogadores - Partida {match_id}")
        table.add_column("Jogador", style="bold")
        table.add_column("Time")
        table.add_column("Min", justify="right")
        table.add_column("Gols", justify="right")
        table.add_column("Assist.", justify="right")
        table.add_column("xG", justify="right")
        table.add_column("xA", justify="right")
        table.add_column("xG+xA", justify="right", style="green")
        table.add_column("Passes", justify="right")
        table.add_column("Finalizacoes", justify="right")

        for _, row in metrics_df.iterrows():
            table.add_row(
                str(row.get("player_name", "")),
                str(row.get("team", "")),
                str(row.get("minutes_played", 0)),
                str(int(row.get("goals", 0) or 0)),
                str(int(row.get("assists", 0) or 0)),
                f"{float(row.get('xg', 0) or 0):.2f}",
                f"{float(row.get('xa', 0) or 0):.2f}",
                f"{row['xg_contribution']:.2f}",
                str(int(row.get("passes", 0) or 0)),
                str(int(row.get("shots", 0) or 0)),
            )

        console.print(table)

        # Display top passing partnerships
        try:
            if "graph" not in dir():
                graph, _ = network_analysis.build_pass_network(match_id)
            partnerships = network_analysis.identify_key_partnerships(graph, top_n=5)

            if partnerships:
                partner_table = Table(title="Principais Parcerias de Passe")
                partner_table.add_column("Passador", style="bold")
                partner_table.add_column("Receptor", style="bold")
                partner_table.add_column("Passes", justify="right")
                partner_table.add_column("% do Total", justify="right")

                for p in partnerships:
                    partner_table.add_row(
                        p["passer"],
                        p["receiver"],
                        str(int(p["weight"])),
                        f"{p['normalized_weight'] * 100:.1f}%",
                    )

                console.print(partner_table)
        except Exception as exc:
            console.print(f"[yellow]Aviso: nao foi possivel exibir parcerias de passe: {exc}[/yellow]")

        # v0.2.0 — Pressing metrics display
        try:
            pressing_rows = (
                session.query(db.PressingMetrics)
                .filter_by(match_id=match_id)
                .all()
            )
            if pressing_rows:
                press_table = Table(title="Metricas de Pressing")
                press_table.add_column("Time", style="bold")
                press_table.add_column("PPDA", justify="right")
                press_table.add_column("Sucesso %", justify="right")
                press_table.add_column("Counter-press %", justify="right")
                press_table.add_column("High Turnovers", justify="right")

                for pr in pressing_rows:
                    press_table.add_row(
                        str(pr.team),
                        f"{pr.ppda:.1f}" if pr.ppda else "-",
                        f"{pr.pressing_success_rate:.0f}%" if pr.pressing_success_rate else "-",
                        f"{pr.counter_pressing_fraction:.0f}%" if pr.counter_pressing_fraction else "-",
                        str(pr.high_turnovers or 0),
                    )
                console.print(press_table)
        except Exception:
            pass

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao analisar partida: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        session.close()


@app.command("analyze-season")
def analyze_season(
    competition: str = typer.Argument(..., help="Nome da competicao"),
    season: str = typer.Argument(..., help="Temporada (ex: 2023/2024)"),
    team: str = typer.Argument(..., help="Nome do time"),
    refresh: bool = typer.Option(False, "--refresh", help="Forca reprocessamento de todas as partidas"),
) -> None:
    """Processa todas as partidas de um time em uma competicao/temporada."""
    session = _get_session()

    try:
        from statsbombpy import sb
        import pandas as pd

        with console.status("[bold green]Buscando competicoes e partidas..."):
            comps = sb.competitions()
            comp_match = comps[
                (comps["competition_name"] == competition)
                & (comps["season_name"] == season)
            ]

            if comp_match.empty:
                console.print(
                    f"[red]Erro: competicao '{competition}' temporada '{season}' nao encontrada.[/red]"
                )
                raise typer.Exit(1)

            comp_row = comp_match.iloc[0]
            matches = sb.matches(
                competition_id=int(comp_row["competition_id"]),
                season_id=int(comp_row["season_id"]),
            )

        team_matches = matches[
            (matches["home_team"] == team) | (matches["away_team"] == team)
        ]

        if team_matches.empty:
            console.print(f"[red]Erro: nenhuma partida encontrada para o time '{team}'.[/red]")
            raise typer.Exit(1)

        console.print(
            f"[cyan]Encontradas {len(team_matches)} partidas para {team} em {competition} {season}.[/cyan]"
        )

        all_metrics = []

        with Progress() as progress:
            task = progress.add_task(
                "[green]Processando partidas...",
                total=len(team_matches),
            )

            for _, match_row in team_matches.iterrows():
                mid = int(match_row["match_id"])

                if not refresh and db.match_exists(session, mid):
                    rows = (
                        session.query(db.PlayerMatchMetrics)
                        .filter_by(match_id=mid)
                        .all()
                    )
                    columns = [c.key for c in db.PlayerMatchMetrics.__table__.columns]
                    match_metrics = pd.DataFrame(
                        [{col: getattr(r, col) for col in columns} for r in rows]
                    )
                else:
                    try:
                        match_metrics = player_metrics.extract_match_metrics(mid)
                        if not match_metrics.empty:
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
                            db.upsert_match(session, match_info)
                            db.upsert_player_metrics(session, match_metrics, mid)

                            graph, edges_df = network_analysis.build_pass_network(mid, team=team)
                            edge_features = network_analysis.compute_edge_features(graph)
                            for idx, erow in edges_df.iterrows():
                                key = (erow["passer_id"], erow["receiver_id"])
                                if key in edge_features:
                                    edges_df.at[idx, "features"] = edge_features[key]
                            db.upsert_pass_network(session, edges_df, mid)

                            # v0.2.0 — Pressing per match
                            try:
                                pressing_df = pressing.compute_match_pressing(mid)
                                if not pressing_df.empty:
                                    db.upsert_pressing_metrics(session, pressing_df, mid)
                            except Exception:
                                pass
                    except Exception as exc:
                        console.print(f"[yellow]Aviso: falha na partida {mid}: {exc}[/yellow]")
                        progress.advance(task)
                        continue

                if not match_metrics.empty:
                    all_metrics.append(match_metrics)

                progress.advance(task)

        if not all_metrics:
            console.print("[red]Erro: nenhuma metrica extraida.[/red]")
            raise typer.Exit(1)

        combined = pd.concat(all_metrics, ignore_index=True)

        # Filter for the target team only
        team_data = combined[combined["team"] == team]

        if team_data.empty:
            console.print("[red]Erro: nenhuma metrica encontrada para o time especificado.[/red]")
            raise typer.Exit(1)

        # Aggregate player stats
        numeric_cols = team_data.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("player_id", "match_id")]

        agg_stats = (
            team_data.groupby(["player_id", "player_name"])[numeric_cols]
            .sum()
            .reset_index()
        )
        agg_stats["partidas"] = team_data.groupby("player_id").size().values
        agg_stats = agg_stats.sort_values("xg", ascending=False)

        # Display aggregated stats table
        agg_table = Table(title=f"Estatisticas Agregadas - {team} ({competition} {season})")
        agg_table.add_column("Jogador", style="bold")
        agg_table.add_column("Partidas", justify="right")
        agg_table.add_column("Min", justify="right")
        agg_table.add_column("Gols", justify="right")
        agg_table.add_column("Assist.", justify="right")
        agg_table.add_column("xG", justify="right")
        agg_table.add_column("xA", justify="right")
        agg_table.add_column("Passes", justify="right")
        agg_table.add_column("Dribles", justify="right")
        agg_table.add_column("Desarmes", justify="right")

        for _, row in agg_stats.iterrows():
            agg_table.add_row(
                str(row["player_name"]),
                str(int(row.get("partidas", 0))),
                str(round(row.get("minutes_played", 0) or 0, 1)),
                str(int(row.get("goals", 0) or 0)),
                str(int(row.get("assists", 0) or 0)),
                f"{float(row.get('xg', 0) or 0):.2f}",
                f"{float(row.get('xa', 0) or 0):.2f}",
                str(int(row.get("passes", 0) or 0)),
                str(int(row.get("dribbles_completed", 0) or 0)),
                str(int(row.get("tackles", 0) or 0)),
            )

        console.print(agg_table)

        # Generate embeddings and compute RAPM
        try:
            from football_moneyball import player_embeddings

            with console.status("[bold green]Gerando embeddings dos jogadores..."):
                profiles = player_embeddings.build_player_profiles(session, competition, season)
                if not profiles.empty:
                    emb_df, pca = player_embeddings.generate_embeddings(profiles)
                    emb_df = player_embeddings.cluster_players(emb_df, pca=pca)
                    emb_df["season"] = season
                    emb_df["competition"] = competition
                    db.upsert_embeddings(session, emb_df)
                    console.print("[green]Embeddings gerados e salvos com sucesso.[/green]")
        except ImportError:
            console.print("[yellow]Aviso: modulo player_embeddings nao disponivel. Embeddings nao gerados.[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Aviso: falha ao gerar embeddings: {exc}[/yellow]")

        try:
            from football_moneyball import rapm as rapm_mod

            with console.status("[bold green]Calculando RAPM..."):
                rapm_results = rapm_mod.compute_season_rapm(session, competition, season)
                if not rapm_results.empty:
                    console.print("[green]RAPM calculado com sucesso.[/green]")
        except ImportError:
            console.print("[yellow]Aviso: modulo RAPM nao disponivel.[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Aviso: falha ao calcular RAPM: {exc}[/yellow]")

        # Season summary panel
        total_goals = int(agg_stats["goals"].sum()) if "goals" in agg_stats.columns else 0
        total_xg = float(agg_stats["xg"].sum()) if "xg" in agg_stats.columns else 0.0
        total_matches = len(team_matches)
        total_players = len(agg_stats)

        summary_text = (
            f"[bold]{team}[/bold] - {competition} {season}\n\n"
            f"Partidas: {total_matches}\n"
            f"Jogadores utilizados: {total_players}\n"
            f"Gols totais: {total_goals}\n"
            f"xG total: {total_xg:.2f}\n"
        )

        console.print(Panel(summary_text, title="Resumo da Temporada", border_style="green"))

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao analisar temporada: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        session.close()


@app.command("compare-players")
def compare_players(
    player_a: str = typer.Argument(..., help="Nome do primeiro jogador"),
    player_b: str = typer.Argument(..., help="Nome do segundo jogador"),
    season: Optional[str] = typer.Option(None, "--season", help="Filtrar por temporada"),
) -> None:
    """Compara metricas de dois jogadores lado a lado."""
    session = _get_session()

    try:
        import pandas as pd
        import numpy as np

        metrics_a = db.get_player_metrics(session, player_a, season)
        metrics_b = db.get_player_metrics(session, player_b, season)

        if metrics_a.empty:
            console.print(f"[red]Erro: jogador '{player_a}' nao encontrado no banco de dados.[/red]")
            raise typer.Exit(1)
        if metrics_b.empty:
            console.print(f"[red]Erro: jogador '{player_b}' nao encontrado no banco de dados.[/red]")
            raise typer.Exit(1)

        # Aggregate metrics across matches
        numeric_cols = metrics_a.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("player_id", "match_id")]

        agg_a = metrics_a[numeric_cols].sum()
        agg_a["partidas"] = len(metrics_a)
        agg_b = metrics_b[numeric_cols].sum()
        agg_b["partidas"] = len(metrics_b)

        # Display comparison table
        compare_table = Table(title=f"Comparacao: {player_a} vs {player_b}")
        compare_table.add_column("Metrica", style="bold")
        compare_table.add_column(player_a, justify="right")
        compare_table.add_column(player_b, justify="right")

        display_metrics = [
            ("Partidas", "partidas"),
            ("Minutos", "minutes_played"),
            ("Gols", "goals"),
            ("Assistencias", "assists"),
            ("xG", "xg"),
            ("xA", "xa"),
            ("Big Chances", "big_chances"),
            ("Passes", "passes"),
            ("Passes Completados", "passes_completed"),
            ("Passes Curtos", "passes_short"),
            ("Passes Medios", "passes_medium"),
            ("Passes Longos", "passes_long"),
            ("Passes sob Pressao", "passes_under_pressure"),
            ("Passes Progressivos", "progressive_passes"),
            ("Recepcoes Progressivas", "progressive_receptions"),
            ("Passes Decisivos", "key_passes"),
            ("Mudancas de Jogo", "switches_of_play"),
            ("Finalizacoes", "shots"),
            ("Finalizacoes no Alvo", "shots_on_target"),
            ("Dribles Completados", "dribbles_completed"),
            ("Desarmes", "tackles"),
            ("Taxa Sucesso Desarme", "tackle_success_rate"),
            ("Duelos Terrestres", "ground_duels_total"),
            ("Interceptacoes", "interceptions"),
            ("Bloqueios", "blocks"),
            ("Pressoes", "pressures"),
            ("Toques", "touches"),
            ("Conducoes", "carries"),
            ("Conducoes Progressivas", "progressive_carries"),
        ]

        for label, key in display_metrics:
            val_a = agg_a.get(key, 0) or 0
            val_b = agg_b.get(key, 0) or 0

            if key in ("xg", "xa"):
                str_a = f"{float(val_a):.2f}"
                str_b = f"{float(val_b):.2f}"
            elif key == "minutes_played":
                str_a = f"{float(val_a):.1f}"
                str_b = f"{float(val_b):.1f}"
            else:
                str_a = str(int(val_a))
                str_b = str(int(val_b))

            compare_table.add_row(label, str_a, str_b)

        console.print(compare_table)

        # Radar comparison
        try:
            from football_moneyball import viz

            radar_a = {"name": player_a}
            radar_b = {"name": player_b}
            for col in numeric_cols:
                radar_a[col] = float(agg_a.get(col, 0) or 0)
                radar_b[col] = float(agg_b.get(col, 0) or 0)

            viz.plot_radar_comparison(radar_a, radar_b)
            console.print("[green]Grafico radar gerado.[/green]")
        except ImportError:
            console.print("[yellow]Aviso: modulo viz nao disponivel para gerar radar.[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Aviso: falha ao gerar grafico radar: {exc}[/yellow]")

        # Similarity score (cosine distance from embeddings)
        try:
            emb_a = (
                session.query(db.PlayerEmbedding)
                .filter_by(player_name=player_a)
            )
            emb_b = (
                session.query(db.PlayerEmbedding)
                .filter_by(player_name=player_b)
            )

            if season:
                emb_a = emb_a.filter_by(season=season)
                emb_b = emb_b.filter_by(season=season)

            emb_a = emb_a.first()
            emb_b = emb_b.first()

            if emb_a and emb_b and emb_a.embedding is not None and emb_b.embedding is not None:
                vec_a = np.array(emb_a.embedding)
                vec_b = np.array(emb_b.embedding)

                dot = np.dot(vec_a, vec_b)
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)

                if norm_a > 0 and norm_b > 0:
                    cosine_sim = dot / (norm_a * norm_b)
                    cosine_dist = 1 - cosine_sim

                    console.print(
                        Panel(
                            f"Similaridade cosseno: [bold]{cosine_sim:.4f}[/bold]\n"
                            f"Distancia cosseno: [bold]{cosine_dist:.4f}[/bold]",
                            title="Similaridade entre Jogadores",
                            border_style="cyan",
                        )
                    )
            else:
                console.print("[yellow]Aviso: embeddings nao encontrados para calcular similaridade.[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Aviso: falha ao calcular similaridade: {exc}[/yellow]")

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao comparar jogadores: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        session.close()


@app.command("find-similar")
def find_similar(
    player: str = typer.Argument(..., help="Nome do jogador de referencia"),
    season: Optional[str] = typer.Option(None, "--season", help="Filtrar por temporada"),
    limit: int = typer.Option(10, "--limit", help="Numero maximo de resultados"),
) -> None:
    """Busca jogadores similares usando busca vetorial pgvector."""
    session = _get_session()

    try:
        if season is None:
            # Try to get the latest season for this player
            latest = (
                session.query(db.PlayerEmbedding)
                .filter_by(player_name=player)
                .order_by(db.PlayerEmbedding.season.desc())
                .first()
            )
            if latest is None:
                console.print(f"[red]Erro: jogador '{player}' nao encontrado nos embeddings.[/red]")
                raise typer.Exit(1)
            season = latest.season
            console.print(f"[cyan]Usando temporada mais recente: {season}[/cyan]")

        # Try dedicated module first, fall back to db function
        try:
            from football_moneyball import player_embeddings

            similar_df = player_embeddings.find_similar(session, player, season, limit)
        except (ImportError, AttributeError):
            similar_df = db.find_similar_players(session, player, season, limit)

        if similar_df.empty:
            console.print(f"[yellow]Nenhum jogador similar encontrado para '{player}'.[/yellow]")
            raise typer.Exit(0)

        table = Table(title=f"Jogadores Similares a {player} ({season})")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Jogador", style="bold")
        table.add_column("Posicao")
        table.add_column("Distancia", justify="right")
        table.add_column("Arquetipo")

        for idx, row in similar_df.iterrows():
            table.add_row(
                str(idx + 1 if isinstance(idx, int) else idx),
                str(row.get("player_name", "")),
                str(row.get("position_group", row.get("season", ""))),
                f"{float(row.get('distance', 0)):.4f}",
                str(row.get("archetype", "-")),
            )

        console.print(table)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao buscar jogadores similares: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        session.close()


@app.command("recommend")
def recommend(
    profile_path: str = typer.Argument(..., help="Caminho para o arquivo JSON com perfil desejado"),
    season: Optional[str] = typer.Option(None, "--season", help="Filtrar por temporada"),
    limit: int = typer.Option(10, "--limit", help="Numero maximo de recomendacoes"),
) -> None:
    """Recomenda jogadores com base em um perfil de atributos desejados."""
    try:
        profile_file = Path(profile_path)
        if not profile_file.exists():
            console.print(f"[red]Erro: arquivo '{profile_path}' nao encontrado.[/red]")
            raise typer.Exit(1)

        with open(profile_file, "r", encoding="utf-8") as f:
            profile = json.load(f)

        console.print(f"[cyan]Perfil carregado: {len(profile)} atributos definidos.[/cyan]")

    except json.JSONDecodeError as exc:
        console.print(f"[red]Erro: arquivo JSON invalido: {exc}[/red]")
        raise typer.Exit(1)

    session = _get_session()

    try:
        from football_moneyball import player_embeddings

        with console.status("[bold green]Buscando recomendacoes..."):
            recommendations = player_embeddings.recommend_by_profile(
                session, profile, season=season, limit=limit
            )

        if recommendations.empty:
            console.print("[yellow]Nenhuma recomendacao encontrada para o perfil informado.[/yellow]")
            raise typer.Exit(0)

        table = Table(title="Recomendacoes de Jogadores")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Jogador", style="bold")
        table.add_column("Temporada")
        table.add_column("Score", justify="right")
        table.add_column("Arquetipo")

        for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
            table.add_row(
                str(rank),
                str(row.get("player_name", "")),
                str(row.get("season", "")),
                f"{float(row.get('score', row.get('distance', 0))):.4f}",
                str(row.get("archetype", "-")),
            )

        console.print(table)

    except ImportError:
        console.print("[red]Erro: modulo player_embeddings nao disponivel.[/red]")
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao gerar recomendacoes: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        session.close()


@app.command("scout-report")
def scout_report(
    player: str = typer.Argument(..., help="Nome do jogador"),
    season: Optional[str] = typer.Option(None, "--season", help="Filtrar por temporada"),
    team_target: Optional[str] = typer.Option(None, "--team-target", help="Time alvo para analise de compatibilidade"),
    output: Optional[str] = typer.Option(None, "--output", help="Caminho do arquivo de saida (markdown ou JSON)"),
) -> None:
    """Gera relatorio completo de scouting de um jogador."""
    session = _get_session()

    try:
        import numpy as np

        # Fetch player metrics
        metrics_df = db.get_player_metrics(session, player, season)
        if metrics_df.empty:
            console.print(f"[red]Erro: jogador '{player}' nao encontrado no banco de dados.[/red]")
            raise typer.Exit(1)

        # Aggregate metrics
        numeric_cols = metrics_df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("player_id", "match_id")]
        agg = metrics_df[numeric_cols].sum()
        matches_played = len(metrics_df)
        per90 = {k: round(float(v) / (float(agg.get("minutes_played", 1) or 1) / 90), 2) for k, v in agg.items()}

        # Percentiles (compared to all players in the same season)
        percentiles = {}
        try:
            from sqlalchemy import func

            if season:
                all_players_q = (
                    session.query(db.PlayerMatchMetrics)
                    .join(db.Match, db.Match.match_id == db.PlayerMatchMetrics.match_id)
                    .filter(db.Match.season == season)
                )
            else:
                all_players_q = session.query(db.PlayerMatchMetrics)

            import pandas as pd
            all_rows = all_players_q.all()
            columns = [c.key for c in db.PlayerMatchMetrics.__table__.columns]
            all_data = pd.DataFrame([{col: getattr(r, col) for col in columns} for r in all_rows])

            if not all_data.empty:
                all_agg = (
                    all_data.groupby("player_id")[numeric_cols].sum()
                )
                for col in numeric_cols:
                    if col in all_agg.columns and col in agg.index:
                        rank = (all_agg[col] < float(agg[col])).sum()
                        percentiles[col] = round(rank / len(all_agg) * 100, 1)
        except Exception:
            pass

        # Player embedding / archetype
        emb = session.query(db.PlayerEmbedding).filter_by(player_name=player)
        if season:
            emb = emb.filter_by(season=season)
        emb = emb.first()
        archetype = emb.archetype if emb else "N/A"

        # Top 5 similar players
        similar_df = db.find_similar_players(session, player, season or "", limit=5)

        # Build report data
        report = {
            "jogador": player,
            "temporada": season or "Todas",
            "partidas": matches_played,
            "arquetipo": archetype,
            "metricas_totais": {k: float(v) for k, v in agg.items()},
            "metricas_por_90": per90,
            "percentis": percentiles,
            "similares": similar_df.to_dict("records") if not similar_df.empty else [],
        }

        # Compatibility analysis
        if team_target:
            report["time_alvo"] = team_target
            try:
                from football_moneyball import player_embeddings

                compat_df = player_embeddings.find_complementary(
                    session, player, season or "", limit=10
                )
                report["compatibilidade"] = compat_df.to_dict("records") if not compat_df.empty else []
            except (ImportError, AttributeError):
                report["compatibilidade"] = "Analise de compatibilidade nao disponivel."

        # Output handling
        if output:
            output_path = Path(output)

            if output_path.suffix == ".json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            else:
                # Markdown output
                try:
                    from football_moneyball import export

                    full_report = export.generate_scout_report(session, player, season, team_target)
                    export.save_report(full_report, str(output_path), format="markdown")
                except ImportError:
                    # Fallback: write basic markdown
                    lines = [
                        f"# Relatorio de Scouting: {player}",
                        f"\n**Temporada:** {report['temporada']}",
                        f"**Partidas:** {matches_played}",
                        f"**Arquetipo:** {archetype}",
                        "\n## Metricas Totais",
                    ]
                    for k, v in report["metricas_totais"].items():
                        lines.append(f"- {k}: {v}")
                    lines.append("\n## Metricas por 90 minutos")
                    for k, v in report["metricas_por_90"].items():
                        lines.append(f"- {k}: {v}")
                    if percentiles:
                        lines.append("\n## Percentis")
                        for k, v in percentiles.items():
                            lines.append(f"- {k}: {v}%")
                    if report["similares"]:
                        lines.append("\n## Jogadores Similares")
                        for s in report["similares"]:
                            lines.append(f"- {s.get('player_name', '')} (distancia: {s.get('distance', '')})")

                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")

            console.print(f"[green]Relatorio salvo em: {output_path}[/green]")
        else:
            # Display in terminal
            header_text = (
                f"[bold]{player}[/bold]\n"
                f"Temporada: {report['temporada']}\n"
                f"Partidas: {matches_played}\n"
                f"Arquetipo: [bold cyan]{archetype}[/bold cyan]"
            )
            console.print(Panel(header_text, title="Relatorio de Scouting", border_style="blue"))

            # Metrics table
            metrics_table = Table(title="Metricas")
            metrics_table.add_column("Metrica", style="bold")
            metrics_table.add_column("Total", justify="right")
            metrics_table.add_column("Por 90 min", justify="right")
            metrics_table.add_column("Percentil", justify="right")

            display_keys = [
                ("Gols", "goals"),
                ("Assistencias", "assists"),
                ("xG", "xg"),
                ("xA", "xa"),
                ("Big Chances", "big_chances"),
                ("Passes", "passes"),
                ("Passes Completados", "passes_completed"),
                ("Passes Curtos", "passes_short"),
                ("Passes Longos", "passes_long"),
                ("Passes sob Pressao", "passes_under_pressure"),
                ("Passes Progressivos", "progressive_passes"),
                ("Recepcoes Progressivas", "progressive_receptions"),
                ("Passes Decisivos", "key_passes"),
                ("Mudancas de Jogo", "switches_of_play"),
                ("Finalizacoes", "shots"),
                ("Dribles Completados", "dribbles_completed"),
                ("Desarmes", "tackles"),
                ("Taxa Sucesso Desarme %", "tackle_success_rate"),
                ("Duelos Terrestres", "ground_duels_total"),
                ("Interceptacoes", "interceptions"),
                ("Pressoes", "pressures"),
                ("Conducoes Progressivas", "progressive_carries"),
                ("xT Gerado", "xt_generated"),
                ("VAEP Gerado", "vaep_generated"),
            ]

            for label, key in display_keys:
                total_val = report["metricas_totais"].get(key, 0)
                p90_val = report["metricas_por_90"].get(key, 0)
                pct_val = percentiles.get(key, "-")

                if key in ("xg", "xa"):
                    total_str = f"{total_val:.2f}"
                    p90_str = f"{p90_val:.2f}"
                else:
                    total_str = str(int(total_val))
                    p90_str = f"{p90_val:.2f}"

                pct_str = f"{pct_val}%" if isinstance(pct_val, (int, float)) else str(pct_val)

                metrics_table.add_row(label, total_str, p90_str, pct_str)

            console.print(metrics_table)

            # Similar players
            if not similar_df.empty:
                sim_table = Table(title="Top 5 Jogadores Similares")
                sim_table.add_column("Jogador", style="bold")
                sim_table.add_column("Distancia", justify="right")
                sim_table.add_column("Arquetipo")

                for _, srow in similar_df.iterrows():
                    sim_table.add_row(
                        str(srow.get("player_name", "")),
                        f"{float(srow.get('distance', 0)):.4f}",
                        str(srow.get("archetype", "-")),
                    )

                console.print(sim_table)

            # Compatibility
            if team_target and "compatibilidade" in report:
                console.print(
                    Panel(
                        str(report["compatibilidade"]),
                        title=f"Compatibilidade com {team_target}",
                        border_style="magenta",
                    )
                )

            # Generate radar
            try:
                from football_moneyball import viz

                radar_data = {"name": player}
                for k, v in percentiles.items():
                    radar_data[k] = v
                viz.plot_radar_comparison(radar_data, radar_data)
            except (ImportError, AttributeError):
                pass

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao gerar relatorio de scouting: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        session.close()


@app.command("list-competitions")
def list_competitions() -> None:
    """Lista as competicoes disponiveis nos dados abertos do StatsBomb."""
    try:
        with console.status("[bold green]Buscando competicoes..."):
            comps = player_metrics.get_free_competitions()

        if comps.empty:
            console.print("[yellow]Nenhuma competicao encontrada.[/yellow]")
            raise typer.Exit(0)

        table = Table(title="Competicoes Disponiveis (StatsBomb Open Data)")
        table.add_column("ID Comp.", justify="right", style="dim")
        table.add_column("Competicao", style="bold")
        table.add_column("ID Temp.", justify="right", style="dim")
        table.add_column("Temporada")
        table.add_column("Pais")

        for _, row in comps.iterrows():
            table.add_row(
                str(int(row.get("competition_id", 0))),
                str(row.get("competition_name", "")),
                str(int(row.get("season_id", 0))),
                str(row.get("season_name", "")),
                str(row.get("country_name", "")),
            )

        console.print(table)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao listar competicoes: {exc}[/red]")
        raise typer.Exit(1)


@app.command("list-matches")
def list_matches(
    competition_id: int = typer.Argument(..., help="ID da competicao StatsBomb"),
    season_id: int = typer.Argument(..., help="ID da temporada StatsBomb"),
) -> None:
    """Lista as partidas de uma competicao/temporada."""
    try:
        with console.status("[bold green]Buscando partidas..."):
            matches = player_metrics.get_competition_matches(competition_id, season_id)

        if matches.empty:
            console.print("[yellow]Nenhuma partida encontrada.[/yellow]")
            raise typer.Exit(0)

        table = Table(title=f"Partidas - Competicao {competition_id} / Temporada {season_id}")
        table.add_column("ID", justify="right", style="dim")
        table.add_column("Data")
        table.add_column("Mandante", style="bold")
        table.add_column("Placar", justify="center")
        table.add_column("Visitante", style="bold")

        for _, row in matches.iterrows():
            home_score = row.get("home_score", "")
            away_score = row.get("away_score", "")
            score_str = f"{home_score} x {away_score}" if home_score != "" else "-"

            table.add_row(
                str(int(row.get("match_id", 0))),
                str(row.get("match_date", "")),
                str(row.get("home_team", "")),
                score_str,
                str(row.get("away_team", "")),
            )

        console.print(table)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao listar partidas: {exc}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
