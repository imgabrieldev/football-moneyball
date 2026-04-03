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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
