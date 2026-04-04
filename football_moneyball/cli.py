"""CLI Typer para Football Moneyball Analytics.

Interface de linha de comando com saida Rich para analise de futebol.
Camada fina: argument parsing (Typer) + output formatting (Rich).
Toda logica de negocio e delegada para use cases e adapters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich import print as rprint

from football_moneyball.config import get_provider, get_repository, get_visualizer
from football_moneyball.use_cases.analyze_match import AnalyzeMatch
from football_moneyball.use_cases.analyze_season import AnalyzeSeason
from football_moneyball.use_cases.compare_players import ComparePlayers
from football_moneyball.use_cases.find_similar import FindSimilar
from football_moneyball.use_cases.generate_report import GenerateReport

# ---------------------------------------------------------------------------
# App & globals
# ---------------------------------------------------------------------------

app = typer.Typer(name="moneyball", help="Football Moneyball Analytics CLI")

console = Console()

_provider_name: str = "statsbomb"


@app.callback()
def main(
    provider: str = typer.Option(
        "statsbomb", help="Data provider (statsbomb/sofascore)"
    ),
) -> None:
    """Football Moneyball Analytics — CLI de analytics de futebol."""
    global _provider_name
    _provider_name = provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_repo():
    """Retorna um MatchRepository conectado ao banco de dados."""
    try:
        return get_repository()
    except Exception as exc:
        console.print(f"[red]Erro ao conectar ao banco de dados: {exc}[/red]")
        raise typer.Exit(1)


def _get_provider():
    """Retorna o DataProvider configurado."""
    try:
        return get_provider(_provider_name)
    except Exception as exc:
        console.print(f"[red]Erro ao criar provider '{_provider_name}': {exc}[/red]")
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
    provider = _get_provider()
    repo = _get_repo()

    try:
        use_case = AnalyzeMatch(provider, repo)

        with console.status("[bold green]Analisando partida..."):
            result = use_case.execute(match_id, refresh)

        if "error" in result:
            console.print(f"[red]Erro: {result['error']}[/red]")
            raise typer.Exit(1)

        metrics_df = result["metrics_df"]

        if metrics_df.empty:
            console.print(f"[red]Erro: nenhum dado encontrado para a partida {match_id}.[/red]")
            raise typer.Exit(1)

        if result.get("from_cache"):
            console.print(f"[cyan]Carregando partida {match_id} do banco de dados...[/cyan]")
        else:
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
        partnerships = result.get("partnerships")
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

        # Pressing metrics display
        pressing_df = result.get("pressing_df")
        try:
            if pressing_df is not None and not pressing_df.empty:
                press_table = Table(title="Metricas de Pressing")
                press_table.add_column("Time", style="bold")
                press_table.add_column("PPDA", justify="right")
                press_table.add_column("Sucesso %", justify="right")
                press_table.add_column("Counter-press %", justify="right")
                press_table.add_column("High Turnovers", justify="right")

                for _, pr in pressing_df.iterrows():
                    press_table.add_row(
                        str(pr.get("team", "")),
                        f"{float(pr.get('ppda', 0)):.1f}" if pr.get("ppda") else "-",
                        f"{float(pr.get('pressing_success_rate', 0)):.0f}%" if pr.get("pressing_success_rate") else "-",
                        f"{float(pr.get('counter_pressing_fraction', 0)):.0f}%" if pr.get("counter_pressing_fraction") else "-",
                        str(int(pr.get("high_turnovers", 0) or 0)),
                    )
                console.print(press_table)
            elif result.get("from_cache"):
                # When loaded from cache, try fetching pressing from repo
                pressing_rows = repo.get_pressing_metrics_for_match(match_id)
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
        repo.close()


@app.command("analyze-season")
def analyze_season(
    competition: str = typer.Argument(..., help="Nome da competicao"),
    season: str = typer.Argument(..., help="Temporada (ex: 2023/2024)"),
    team: str = typer.Argument(..., help="Nome do time"),
    refresh: bool = typer.Option(False, "--refresh", help="Forca reprocessamento de todas as partidas"),
) -> None:
    """Processa todas as partidas de um time em uma competicao/temporada."""
    provider = _get_provider()
    repo = _get_repo()

    try:
        # Resolve competition_id and season_id
        with console.status("[bold green]Buscando competicoes e partidas..."):
            comps = provider.get_competitions()
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
            competition_id = int(comp_row["competition_id"])
            season_id = int(comp_row["season_id"])

        use_case = AnalyzeSeason(provider, repo)

        # Progress callback for Rich progress bar
        progress_ctx = Progress()
        task_id = None

        def on_progress(i, total, match_row):
            nonlocal task_id
            if task_id is None:
                task_id = progress_ctx.add_task(
                    "[green]Processando partidas...", total=total
                )
            progress_ctx.advance(task_id)

        console.print(
            f"[cyan]Processando partidas para {team} em {competition} {season}...[/cyan]"
        )

        with progress_ctx:
            # Initialize task before execute so the bar shows
            result = {}

            def on_progress_with_init(i, total, match_row):
                nonlocal task_id
                if task_id is None:
                    task_id = progress_ctx.add_task(
                        "[green]Processando partidas...", total=total
                    )
                progress_ctx.advance(task_id)

            result = use_case.execute(
                competition=competition,
                season=season,
                team=team,
                competition_id=competition_id,
                season_id=season_id,
                refresh=refresh,
                on_progress=on_progress_with_init,
            )

        if "error" in result:
            console.print(f"[red]Erro: {result['error']}[/red]")
            raise typer.Exit(1)

        agg_stats = result["agg_stats"]

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

        # Generate embeddings
        try:
            with console.status("[bold green]Gerando embeddings dos jogadores..."):
                if use_case.generate_embeddings(competition, season):
                    console.print("[green]Embeddings gerados e salvos com sucesso.[/green]")
        except Exception as exc:
            console.print(f"[yellow]Aviso: falha ao gerar embeddings: {exc}[/yellow]")

        # Compute RAPM
        try:
            with console.status("[bold green]Calculando RAPM..."):
                if use_case.compute_rapm(competition, season):
                    console.print("[green]RAPM calculado com sucesso.[/green]")
        except Exception as exc:
            console.print(f"[yellow]Aviso: falha ao calcular RAPM: {exc}[/yellow]")

        # Season summary panel
        total_goals = int(agg_stats["goals"].sum()) if "goals" in agg_stats.columns else 0
        total_xg = float(agg_stats["xg"].sum()) if "xg" in agg_stats.columns else 0.0
        total_matches = result["team_matches_count"]
        total_players = result["total_players"]

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
        repo.close()


@app.command("compare-players")
def compare_players(
    player_a: str = typer.Argument(..., help="Nome do primeiro jogador"),
    player_b: str = typer.Argument(..., help="Nome do segundo jogador"),
    season: Optional[str] = typer.Option(None, "--season", help="Filtrar por temporada"),
) -> None:
    """Compara metricas de dois jogadores lado a lado."""
    repo = _get_repo()

    try:
        use_case = ComparePlayers(repo)
        result = use_case.execute(player_a, player_b, season)

        if "error" in result:
            console.print(f"[red]Erro: {result['error']}[/red]")
            raise typer.Exit(1)

        agg_a = result["agg_a"]
        agg_b = result["agg_b"]

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
            viz = get_visualizer()
            numeric_cols = result["metrics_a_df"].select_dtypes(include="number").columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in ("player_id", "match_id")]

            radar_a = {"name": player_a}
            radar_b = {"name": player_b}
            for col in numeric_cols:
                radar_a[col] = float(agg_a.get(col, 0) or 0)
                radar_b[col] = float(agg_b.get(col, 0) or 0)

            viz.plot_radar_comparison(radar_a, radar_b)
            console.print("[green]Grafico radar gerado.[/green]")
        except Exception as exc:
            console.print(f"[yellow]Aviso: falha ao gerar grafico radar: {exc}[/yellow]")

        # Similarity score
        similarity = result.get("similarity")
        if similarity is not None:
            cosine_dist = 1 - similarity
            console.print(
                Panel(
                    f"Similaridade cosseno: [bold]{similarity:.4f}[/bold]\n"
                    f"Distancia cosseno: [bold]{cosine_dist:.4f}[/bold]",
                    title="Similaridade entre Jogadores",
                    border_style="cyan",
                )
            )
        else:
            console.print("[yellow]Aviso: embeddings nao encontrados para calcular similaridade.[/yellow]")

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao comparar jogadores: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("find-similar")
def find_similar(
    player: str = typer.Argument(..., help="Nome do jogador de referencia"),
    season: Optional[str] = typer.Option(None, "--season", help="Filtrar por temporada"),
    limit: int = typer.Option(10, "--limit", help="Numero maximo de resultados"),
) -> None:
    """Busca jogadores similares usando busca vetorial pgvector."""
    repo = _get_repo()

    try:
        use_case = FindSimilar(repo)
        result = use_case.execute(player, season, limit)

        if "error" in result:
            console.print(f"[red]Erro: {result['error']}[/red]")
            raise typer.Exit(1)

        similar_df = result["similar_df"]
        resolved_season = result["season"]

        if resolved_season != season and season is None:
            console.print(f"[cyan]Usando temporada mais recente: {resolved_season}[/cyan]")

        if similar_df.empty:
            console.print(f"[yellow]Nenhum jogador similar encontrado para '{player}'.[/yellow]")
            raise typer.Exit(0)

        table = Table(title=f"Jogadores Similares a {player} ({resolved_season})")
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
        repo.close()


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

    repo = _get_repo()

    try:
        from football_moneyball.domain import embeddings as emb_mod

        with console.status("[bold green]Buscando recomendacoes..."):
            # Build synthetic embedding from profile using PCA/scaler
            embedding = emb_mod.profile_to_embedding(profile)
            recommendations = repo.recommend_by_profile(
                embedding=embedding,
                season=season or "",
                limit=limit,
                position_group=profile.get("position_group"),
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
        # Fallback: use legacy player_embeddings module
        try:
            from football_moneyball import player_embeddings

            with console.status("[bold green]Buscando recomendacoes..."):
                recommendations = player_embeddings.recommend_by_profile(
                    repo.session, profile, season=season, limit=limit
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
            console.print("[red]Erro: modulo de embeddings nao disponivel.[/red]")
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao gerar recomendacoes: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("scout-report")
def scout_report(
    player: str = typer.Argument(..., help="Nome do jogador"),
    season: Optional[str] = typer.Option(None, "--season", help="Filtrar por temporada"),
    team_target: Optional[str] = typer.Option(None, "--team-target", help="Time alvo para analise de compatibilidade"),
    output: Optional[str] = typer.Option(None, "--output", help="Caminho do arquivo de saida (markdown ou JSON)"),
) -> None:
    """Gera relatorio completo de scouting de um jogador."""
    repo = _get_repo()

    try:
        use_case = GenerateReport(repo)
        report = use_case.execute(player, season, team_target)

        if "error" in report:
            console.print(f"[red]Erro: {report['error']}[/red]")
            raise typer.Exit(1)

        percentiles = report.get("percentis", {})
        similar_list = report.get("similares", [])

        import pandas as pd
        similar_df = pd.DataFrame(similar_list) if similar_list else pd.DataFrame()

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

                    full_report = export.generate_scout_report(
                        repo.session, player, season, team_target
                    )
                    export.save_report(full_report, str(output_path), format="markdown")
                except ImportError:
                    # Fallback: write basic markdown
                    lines = [
                        f"# Relatorio de Scouting: {player}",
                        f"\n**Temporada:** {report['temporada']}",
                        f"**Partidas:** {report['partidas']}",
                        f"**Arquetipo:** {report['arquetipo']}",
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
                f"Partidas: {report['partidas']}\n"
                f"Arquetipo: [bold cyan]{report['arquetipo']}[/bold cyan]"
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
                viz = get_visualizer()
                radar_data = {"name": player}
                for k, v in percentiles.items():
                    radar_data[k] = v
                viz.plot_radar_comparison(radar_data, radar_data)
            except Exception:
                pass

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao gerar relatorio de scouting: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("list-competitions")
def list_competitions() -> None:
    """Lista as competicoes disponiveis nos dados abertos do StatsBomb."""
    try:
        provider = _get_provider()

        with console.status("[bold green]Buscando competicoes..."):
            comps = provider.get_competitions()

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
        provider = _get_provider()

        with console.status("[bold green]Buscando partidas..."):
            matches = provider.get_matches(competition_id, season_id)

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
# v0.4.0 — Betting commands
# ---------------------------------------------------------------------------

@app.command("predict")
def predict(
    match_id: int = typer.Argument(..., help="ID da partida"),
    home_team: str = typer.Option(..., "--home", help="Nome do time da casa"),
    away_team: str = typer.Option(..., "--away", help="Nome do time visitante"),
    simulations: int = typer.Option(10_000, "--sims", help="Numero de simulacoes Monte Carlo"),
) -> None:
    """Preve resultado de uma partida via Monte Carlo + Poisson."""
    from football_moneyball.use_cases.predict_match import PredictMatch

    repo = get_repository()
    try:
        with console.status("[bold green]Rodando simulacao Monte Carlo..."):
            result = PredictMatch(repo).execute(match_id, home_team, away_team, simulations)

        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            raise typer.Exit(1)

        # Header
        console.print(Panel(
            f"[bold]{home_team}[/bold] vs [bold]{away_team}[/bold]\n"
            f"xG esperado: {result['home_xg']:.2f} - {result['away_xg']:.2f}\n"
            f"Simulacoes: {result['simulations']:,}",
            title="Previsao Monte Carlo", border_style="cyan",
        ))

        # 1X2 probabilities
        prob_table = Table(title="Probabilidades 1X2")
        prob_table.add_column("Resultado", style="bold")
        prob_table.add_column("Probabilidade", justify="right")
        prob_table.add_column("Odds justo", justify="right")
        for label, key in [("Casa", "home_win_prob"), ("Empate", "draw_prob"), ("Fora", "away_win_prob")]:
            prob = result[key]
            fair_odds = round(1 / prob, 2) if prob > 0 else 0
            prob_table.add_row(label, f"{prob*100:.1f}%", f"{fair_odds:.2f}")
        console.print(prob_table)

        # Markets
        market_table = Table(title="Outros Mercados")
        market_table.add_column("Mercado", style="bold")
        market_table.add_column("Probabilidade", justify="right")
        for label, key in [
            ("Over 0.5 gols", "over_05"), ("Over 1.5 gols", "over_15"),
            ("Over 2.5 gols", "over_25"), ("Over 3.5 gols", "over_35"),
            ("Ambas marcam", "btts_prob"),
        ]:
            prob = result.get(key, 0)
            market_table.add_row(label, f"{prob*100:.1f}%")
        console.print(market_table)

        # Most likely scores
        console.print(f"\n[cyan]Placar mais provavel:[/cyan] [bold]{result['most_likely_score']}[/bold]")
        if result.get("score_matrix"):
            score_table = Table(title="Top 5 Placares")
            score_table.add_column("Placar", style="bold")
            score_table.add_column("Probabilidade", justify="right")
            for score, prob in list(result["score_matrix"].items())[:5]:
                score_table.add_row(score, f"{prob*100:.1f}%")
            console.print(score_table)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro na previsao: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("value-bets")
def value_bets(
    bankroll: float = typer.Option(1000.0, "--bankroll", help="Bankroll atual"),
    min_edge: float = typer.Option(0.03, "--min-edge", help="Edge minimo (0.03 = 3%)"),
) -> None:
    """Busca value bets nas proximas partidas via Betfair Exchange."""
    from football_moneyball.config import get_odds_provider
    from football_moneyball.use_cases.find_value_bets import FindValueBets

    repo = get_repository()
    try:
        odds_provider = get_odds_provider()
        with console.status("[bold green]Buscando odds na Betfair Exchange..."):
            result = FindValueBets(odds_provider, repo).execute(
                bankroll=bankroll, min_edge=min_edge
            )

        if not result.get("value_bets"):
            console.print(f"[yellow]Nenhuma value bet encontrada (edge > {min_edge*100:.0f}%).[/yellow]")
            console.print(f"Partidas analisadas: {result.get('total_matches', 0)}")
            raise typer.Exit(0)

        console.print(Panel(
            f"Partidas analisadas: {result['total_matches']}\n"
            f"Partidas com value: {result['matches_with_value']}\n"
            f"Value bets encontradas: {len(result['value_bets'])}",
            title="Value Bets — Betfair Exchange", border_style="green",
        ))

        vb_table = Table(title="Value Bets Recomendadas")
        vb_table.add_column("Partida", style="bold")
        vb_table.add_column("Mercado")
        vb_table.add_column("Aposta")
        vb_table.add_column("Modelo", justify="right")
        vb_table.add_column("Odds", justify="right")
        vb_table.add_column("Edge", justify="right", style="green")
        vb_table.add_column("EV", justify="right")
        vb_table.add_column("Stake", justify="right")

        for vb in result["value_bets"]:
            vb_table.add_row(
                str(vb.get("match", ""))[:30],
                vb["market"],
                vb["outcome"],
                f"{vb['model_prob']*100:.1f}%",
                f"{vb['best_odds']:.2f}",
                f"+{vb['edge']*100:.1f}%",
                f"{vb.get('ev', 0):.3f}",
                f"R$ {vb.get('stake', 0):.2f}",
            )
        console.print(vb_table)

    except typer.Exit:
        raise
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("[dim]Configure: BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY[/dim]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Erro ao buscar value bets: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("backtest")
def backtest(
    season: str = typer.Option("2026", "--season", help="Temporada"),
    competition: str = typer.Option("Brasileirão Série A", "--competition", help="Competicao"),
    bankroll: float = typer.Option(1000.0, "--bankroll", help="Bankroll inicial"),
    min_edge: float = typer.Option(0.03, "--min-edge", help="Edge minimo (0.03 = 3%)"),
) -> None:
    """Roda backtesting com dados historicos do Brasileirao."""
    from football_moneyball.use_cases.backtest import Backtest

    repo = get_repository()
    try:
        with console.status("[bold green]Rodando backtesting..."):
            result = Backtest(repo).execute(
                competition=competition,
                season=season,
                initial_bankroll=bankroll,
                min_edge=min_edge,
            )

        if "error" in result:
            console.print(f"[yellow]{result['error']}[/yellow]")
            if "predictions" in result:
                console.print(f"Partidas analisadas: {result.get('matches_analyzed', 0)}")
            raise typer.Exit(0)

        # Results panel
        roi_color = "green" if result["roi"] > 0 else "red"
        console.print(Panel(
            f"[bold]Backtesting — {competition} {season}[/bold]\n\n"
            f"Bankroll inicial: R$ {result['initial_bankroll']:.2f}\n"
            f"Bankroll final: R$ {result['final_bankroll']:.2f}\n"
            f"[{roi_color}]ROI: {result['roi']:+.2f}%[/{roi_color}]\n\n"
            f"Partidas analisadas: {result['matches_analyzed']}\n"
            f"Apostas realizadas: {result['bets_placed']}\n"
            f"Apostas ganhas: {result['bets_won']}\n"
            f"Hit rate: {result['hit_rate']:.1f}%\n"
            f"Edge medio: {result['avg_edge']:.2f}%\n"
            f"Odds media: {result['avg_odds']:.2f}\n\n"
            f"Brier score: {result['brier_score']:.4f} (< 0.25 = melhor que aleatorio)\n"
            f"Max drawdown: {result['max_drawdown']:.1f}%\n"
            f"Total apostado: R$ {result['total_staked']:.2f}\n"
            f"Retorno total: R$ {result['total_return']:.2f}",
            title="Resultados do Backtesting", border_style=roi_color,
        ))

        # Top bets
        if result.get("bets"):
            bets_table = Table(title="Ultimas 10 Apostas")
            bets_table.add_column("Partida", style="bold")
            bets_table.add_column("Mercado")
            bets_table.add_column("Aposta")
            bets_table.add_column("Odds", justify="right")
            bets_table.add_column("Edge", justify="right")
            bets_table.add_column("Stake", justify="right")
            bets_table.add_column("Resultado")
            bets_table.add_column("Lucro", justify="right")

            for bet in result["bets"][-10:]:
                won_str = "[green]✓[/green]" if bet["won"] else "[red]✗[/red]"
                profit_str = f"[green]+{bet['profit']:.2f}[/green]" if bet["profit"] > 0 else f"[red]{bet['profit']:.2f}[/red]"
                bets_table.add_row(
                    str(bet["match"])[:30],
                    bet["market"],
                    bet["outcome"],
                    f"{bet['odds']:.2f}",
                    f"{bet['edge']*100:.1f}%",
                    f"R$ {bet['stake']:.2f}",
                    won_str,
                    profit_str,
                )
            console.print(bets_table)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro no backtesting: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("verify")
def verify(
    season: str = typer.Option("2026", "--season", help="Temporada"),
    competition: str = typer.Option("Brasileirão Série A", "--competition", help="Competicao"),
) -> None:
    """Verifica previsoes vs resultados reais."""
    from football_moneyball.use_cases.verify_predictions import VerifyPredictions

    repo = get_repository()
    try:
        with console.status("[bold green]Comparando previsoes com resultados..."):
            result = VerifyPredictions(repo).execute(competition=competition, season=season)

        if "error" in result:
            console.print(f"[yellow]{result['error']}[/yellow]")
            raise typer.Exit(0)

        # Summary
        acc_color = "green" if result["accuracy_1x2"] > 40 else "red"
        brier_color = "green" if result["avg_brier_score"] < 0.25 else "yellow"
        console.print(Panel(
            f"[bold]Verificacao — {competition} {season}[/bold]\n\n"
            f"Partidas verificadas: {result['total_matches']}\n\n"
            f"[{acc_color}]1X2 corretos: {result['correct_1x2']}/{result['total_matches']} "
            f"({result['accuracy_1x2']}%)[/{acc_color}]\n"
            f"Over/Under corretos: {result['correct_over_under']}/{result['total_matches']} "
            f"({result['accuracy_over_under']}%)\n"
            f"[{brier_color}]Brier score: {result['avg_brier_score']:.4f}[/{brier_color}] "
            f"(< 0.25 = melhor que aleatorio)",
            title="Modelo vs Realidade", border_style=acc_color,
        ))

        # Detail table
        detail_table = Table(title="Detalhamento por Partida")
        detail_table.add_column("Partida", style="bold")
        detail_table.add_column("Placar")
        detail_table.add_column("Prev 1X2")
        detail_table.add_column("Real")
        detail_table.add_column("Acertou")
        detail_table.add_column("P(H)", justify="right")
        detail_table.add_column("P(D)", justify="right")
        detail_table.add_column("P(A)", justify="right")
        detail_table.add_column("O/U 2.5")
        detail_table.add_column("Brier", justify="right")

        for p in result["predictions"]:
            correct = "[green]✓[/green]" if p["correct_1x2"] else "[red]✗[/red]"
            ou = "[green]✓[/green]" if p["correct_over"] else "[red]✗[/red]"
            detail_table.add_row(
                str(p["match"])[:28],
                p["score"],
                p["predicted"],
                p["actual"],
                correct,
                f"{p['home_prob']*100:.0f}%",
                f"{p['draw_prob']*100:.0f}%",
                f"{p['away_prob']*100:.0f}%",
                ou,
                f"{p['brier']:.3f}",
            )
        console.print(detail_table)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro na verificacao: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# v0.6.0 — Automation commands
# ---------------------------------------------------------------------------

@app.command("ingest")
def ingest(
    provider_name: str = typer.Option("sofascore", "--provider", help="Data provider"),
    competition: str = typer.Option("Brasileirão Série A", "--competition"),
    season: str = typer.Option("2026", "--season"),
) -> None:
    """Ingere partidas novas do provider (delta — so jogos que faltam)."""
    from football_moneyball.use_cases.ingest_matches import IngestMatches

    repo = get_repository()
    try:
        provider = get_provider(provider_name)
        with console.status(f"[bold green]Ingerindo de {provider_name}..."):
            result = IngestMatches(provider, repo).execute(competition, season)

        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            raise typer.Exit(1)

        console.print(Panel(
            f"Ingeridos: {result['ingested']}\n"
            f"Ja existiam: {result['skipped']}\n"
            f"Erros: {result['errors']}\n"
            f"Total no provider: {result['total']}",
            title=f"Ingestao — {provider_name}", border_style="green",
        ))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro na ingestao: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("snapshot-odds")
def snapshot_odds() -> None:
    """Busca odds atuais e salva no PostgreSQL."""
    from football_moneyball.use_cases.snapshot_odds import SnapshotOdds
    from football_moneyball.config import get_odds_provider

    repo = get_repository()
    try:
        odds_provider = get_odds_provider()
        odds_provider.repo = repo
        with console.status("[bold green]Buscando odds..."):
            result = SnapshotOdds(odds_provider, repo).execute()

        if "error" in result:
            console.print(f"[yellow]{result['error']}[/yellow]")
            raise typer.Exit(0)

        console.print(Panel(
            f"Partidas: {result['matches']}\n"
            f"Casas de apostas: {result['bookmakers']}\n"
            f"Total de odds: {result['total_odds']}",
            title="Snapshot de Odds", border_style="green",
        ))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao buscar odds: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("predict-all")
def predict_all_cmd(
    competition: str = typer.Option("Brasileirão Série A", "--competition"),
    season: str = typer.Option("2026", "--season"),
) -> None:
    """Preve todos os jogos pendentes da rodada."""
    from football_moneyball.use_cases.predict_all import PredictAll

    repo = get_repository()
    try:
        with console.status("[bold green]Prevendo partidas..."):
            result = PredictAll(repo).execute(competition, season)

        if "error" in result:
            console.print(f"[yellow]{result['error']}[/yellow]")
            raise typer.Exit(0)

        preds = result.get("predictions", [])
        if not preds:
            console.print("[yellow]Nenhuma previsao gerada.[/yellow]")
            raise typer.Exit(0)

        table = Table(title=f"Previsoes — {result['total']} partidas")
        table.add_column("Partida", style="bold")
        table.add_column("xG H", justify="right")
        table.add_column("xG A", justify="right")
        table.add_column("P(H)", justify="right")
        table.add_column("P(D)", justify="right")
        table.add_column("P(A)", justify="right")
        table.add_column("O/U 2.5", justify="right")
        table.add_column("Placar", justify="center")

        for p in preds:
            table.add_row(
                f"{p.get('home_team', '')[:15]} vs {p.get('away_team', '')[:15]}",
                f"{p['home_xg']:.2f}",
                f"{p['away_xg']:.2f}",
                f"{p['home_win_prob']*100:.0f}%",
                f"{p['draw_prob']*100:.0f}%",
                f"{p['away_win_prob']*100:.0f}%",
                f"O {p['over_25']*100:.0f}%",
                p.get("most_likely_score", ""),
            )
        console.print(table)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro nas previsoes: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# v0.9.0 — Track Record commands
# ---------------------------------------------------------------------------

@app.command("resolve")
def resolve_cmd() -> None:
    """Resolve previsoes pendentes com resultados reais."""
    from football_moneyball.use_cases.resolve_predictions import ResolvePredictions

    repo = get_repository()
    try:
        with console.status("[bold green]Resolvendo previsoes..."):
            result = ResolvePredictions(repo).execute()

        console.print(Panel(
            f"Resolvidos: {result['resolved']}\n"
            f"Pendentes: {result['still_pending']}\n"
            f"Erros: {result.get('errors', 0)}",
            title="Resolve", border_style="green",
        ))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao resolver previsoes: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("track-record")
def track_record_cmd() -> None:
    """Mostra track record do modelo."""
    from football_moneyball.domain.track_record import calculate_track_record

    repo = get_repository()
    try:
        preds = repo.get_prediction_history()
        tr = calculate_track_record(preds)

        # Summary panel
        if tr["resolved"] == 0:
            console.print(Panel(
                f"Total: {tr['total']} previsoes ({tr['pending']} pendentes)\n"
                f"Nenhuma previsao resolvida ainda. Rode [bold]moneyball resolve[/bold] primeiro.",
                title="Track Record", border_style="yellow",
            ))
            return

        acc_color = "green" if tr["accuracy_1x2"] > 40 else "red"
        brier_color = "green" if tr["avg_brier"] < 0.25 else "yellow"
        console.print(Panel(
            f"Total: {tr['total']} previsoes ({tr['resolved']} resolvidas, {tr['pending']} pendentes)\n"
            f"[{acc_color}]Accuracy 1X2: {tr['accuracy_1x2']:.1f}%[/{acc_color}]\n"
            f"Accuracy O/U: {tr['accuracy_over_under']:.1f}%\n"
            f"[{brier_color}]Brier: {tr['avg_brier']:.4f}[/{brier_color}]",
            title="Track Record", border_style="cyan",
        ))

        # By round table
        if tr["by_round"]:
            round_table = Table(title="Acuracia por Rodada")
            round_table.add_column("Rodada", justify="right")
            round_table.add_column("Jogos", justify="right")
            round_table.add_column("1X2 %", justify="right")
            round_table.add_column("O/U %", justify="right")
            round_table.add_column("Brier", justify="right")

            for rd in tr["by_round"]:
                round_table.add_row(
                    str(rd["round"]),
                    str(rd["total"]),
                    f"{rd['accuracy_1x2']:.0f}%",
                    f"{rd['accuracy_ou']:.0f}%",
                    f"{rd['avg_brier']:.4f}",
                )
            console.print(round_table)

        # By team table (top 10 by total)
        if tr["by_team"]:
            sorted_teams = sorted(
                tr["by_team"].items(),
                key=lambda x: x[1]["total"],
                reverse=True,
            )[:10]

            team_table = Table(title="Acuracia por Time (Top 10)")
            team_table.add_column("Time", style="bold")
            team_table.add_column("Jogos", justify="right")
            team_table.add_column("1X2 %", justify="right")

            for team, stats in sorted_teams:
                team_table.add_row(
                    team[:25],
                    str(stats["total"]),
                    f"{stats['accuracy_1x2']:.0f}%",
                )
            console.print(team_table)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Erro ao exibir track record: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
