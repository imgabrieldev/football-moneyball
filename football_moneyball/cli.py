"""Typer CLI for Football Moneyball Analytics.

Command-line interface with Rich output for football analysis.
Thin layer: argument parsing (Typer) + output formatting (Rich).
All business logic is delegated to use cases and adapters.
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
    """Football Moneyball Analytics — football analytics CLI."""
    global _provider_name
    _provider_name = provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_repo():
    """Return a MatchRepository connected to the database."""
    try:
        return get_repository()
    except Exception as exc:
        console.print(f"[red]Error connecting to database: {exc}[/red]")
        raise typer.Exit(1)


def _get_provider():
    """Return the configured DataProvider."""
    try:
        return get_provider(_provider_name)
    except Exception as exc:
        console.print(f"[red]Error creating provider '{_provider_name}': {exc}[/red]")
        raise typer.Exit(1)


def _xg_contribution(row) -> float:
    """Compute the xG contribution (xG + xA) from a DataFrame row."""
    xg = row.get("xg", 0) or 0
    xa = row.get("xa", 0) or 0
    return float(xg) + float(xa)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("analyze-match")
def analyze_match(
    match_id: int = typer.Argument(..., help="StatsBomb match ID"),
    refresh: bool = typer.Option(False, "--refresh", help="Force reprocessing even if already in the database"),
) -> None:
    """Analyze a specific match and display player metrics."""
    provider = _get_provider()
    repo = _get_repo()

    try:
        use_case = AnalyzeMatch(provider, repo)

        with console.status("[bold green]Analyzing match..."):
            result = use_case.execute(match_id, refresh)

        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            raise typer.Exit(1)

        metrics_df = result["metrics_df"]

        if metrics_df.empty:
            console.print(f"[red]Error: no data found for match {match_id}.[/red]")
            raise typer.Exit(1)

        if result.get("from_cache"):
            console.print(f"[cyan]Loading match {match_id} from database...[/cyan]")
        else:
            console.print("[green]Data persisted successfully.[/green]")

        # Display player metrics table sorted by xG contribution
        metrics_df = metrics_df.copy()
        metrics_df["xg_contribution"] = metrics_df.apply(_xg_contribution, axis=1)
        metrics_df = metrics_df.sort_values("xg_contribution", ascending=False)

        table = Table(title=f"Player Metrics - Match {match_id}")
        table.add_column("Player", style="bold")
        table.add_column("Team")
        table.add_column("Min", justify="right")
        table.add_column("Goals", justify="right")
        table.add_column("Assists", justify="right")
        table.add_column("xG", justify="right")
        table.add_column("xA", justify="right")
        table.add_column("xG+xA", justify="right", style="green")
        table.add_column("Passes", justify="right")
        table.add_column("Shots", justify="right")

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
            partner_table = Table(title="Top Passing Partnerships")
            partner_table.add_column("Passer", style="bold")
            partner_table.add_column("Receiver", style="bold")
            partner_table.add_column("Passes", justify="right")
            partner_table.add_column("% of Total", justify="right")

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
                press_table = Table(title="Pressing Metrics")
                press_table.add_column("Team", style="bold")
                press_table.add_column("PPDA", justify="right")
                press_table.add_column("Success %", justify="right")
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
                    press_table = Table(title="Pressing Metrics")
                    press_table.add_column("Team", style="bold")
                    press_table.add_column("PPDA", justify="right")
                    press_table.add_column("Success %", justify="right")
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
        console.print(f"[red]Error analyzing match: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("analyze-season")
def analyze_season(
    competition: str = typer.Argument(..., help="Competition name"),
    season: str = typer.Argument(..., help="Season (e.g. 2023/2024)"),
    team: str = typer.Argument(..., help="Team name"),
    refresh: bool = typer.Option(False, "--refresh", help="Force reprocessing of all matches"),
) -> None:
    """Process all matches for a team in a competition/season."""
    provider = _get_provider()
    repo = _get_repo()

    try:
        # Resolve competition_id and season_id
        with console.status("[bold green]Fetching competitions and matches..."):
            comps = provider.get_competitions()
            comp_match = comps[
                (comps["competition_name"] == competition)
                & (comps["season_name"] == season)
            ]

            if comp_match.empty:
                console.print(
                    f"[red]Error: competition '{competition}' season '{season}' not found.[/red]"
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
                    "[green]Processing matches...", total=total
                )
            progress_ctx.advance(task_id)

        console.print(
            f"[cyan]Processing matches for {team} in {competition} {season}...[/cyan]"
        )

        with progress_ctx:
            # Initialize task before execute so the bar shows
            result = {}

            def on_progress_with_init(i, total, match_row):
                nonlocal task_id
                if task_id is None:
                    task_id = progress_ctx.add_task(
                        "[green]Processing matches...", total=total
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
            console.print(f"[red]Error: {result['error']}[/red]")
            raise typer.Exit(1)

        agg_stats = result["agg_stats"]

        # Display aggregated stats table
        agg_table = Table(title=f"Aggregated Stats - {team} ({competition} {season})")
        agg_table.add_column("Player", style="bold")
        agg_table.add_column("Matches", justify="right")
        agg_table.add_column("Min", justify="right")
        agg_table.add_column("Goals", justify="right")
        agg_table.add_column("Assists", justify="right")
        agg_table.add_column("xG", justify="right")
        agg_table.add_column("xA", justify="right")
        agg_table.add_column("Passes", justify="right")
        agg_table.add_column("Dribbles", justify="right")
        agg_table.add_column("Tackles", justify="right")

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
            with console.status("[bold green]Generating player embeddings..."):
                if use_case.generate_embeddings(competition, season):
                    console.print("[green]Embeddings generated and saved successfully.[/green]")
        except Exception as exc:
            console.print(f"[yellow]Warning: failed to generate embeddings: {exc}[/yellow]")

        # Compute RAPM
        try:
            with console.status("[bold green]Computing RAPM..."):
                if use_case.compute_rapm(competition, season):
                    console.print("[green]RAPM computed successfully.[/green]")
        except Exception as exc:
            console.print(f"[yellow]Warning: failed to compute RAPM: {exc}[/yellow]")

        # Season summary panel
        total_goals = int(agg_stats["goals"].sum()) if "goals" in agg_stats.columns else 0
        total_xg = float(agg_stats["xg"].sum()) if "xg" in agg_stats.columns else 0.0
        total_matches = result["team_matches_count"]
        total_players = result["total_players"]

        summary_text = (
            f"[bold]{team}[/bold] - {competition} {season}\n\n"
            f"Matches: {total_matches}\n"
            f"Players used: {total_players}\n"
            f"Total goals: {total_goals}\n"
            f"Total xG: {total_xg:.2f}\n"
        )

        console.print(Panel(summary_text, title="Season Summary", border_style="green"))

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error analyzing season: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("compare-players")
def compare_players(
    player_a: str = typer.Argument(..., help="First player name"),
    player_b: str = typer.Argument(..., help="Second player name"),
    season: Optional[str] = typer.Option(None, "--season", help="Filter by season"),
) -> None:
    """Compare metrics of two players side by side."""
    repo = _get_repo()

    try:
        use_case = ComparePlayers(repo)
        result = use_case.execute(player_a, player_b, season)

        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            raise typer.Exit(1)

        agg_a = result["agg_a"]
        agg_b = result["agg_b"]

        # Display comparison table
        compare_table = Table(title=f"Comparison: {player_a} vs {player_b}")
        compare_table.add_column("Metric", style="bold")
        compare_table.add_column(player_a, justify="right")
        compare_table.add_column(player_b, justify="right")

        display_metrics = [
            ("Matches", "partidas"),
            ("Minutes", "minutes_played"),
            ("Goals", "goals"),
            ("Assists", "assists"),
            ("xG", "xg"),
            ("xA", "xa"),
            ("Big Chances", "big_chances"),
            ("Passes", "passes"),
            ("Completed Passes", "passes_completed"),
            ("Short Passes", "passes_short"),
            ("Medium Passes", "passes_medium"),
            ("Long Passes", "passes_long"),
            ("Passes Under Pressure", "passes_under_pressure"),
            ("Progressive Passes", "progressive_passes"),
            ("Progressive Receptions", "progressive_receptions"),
            ("Key Passes", "key_passes"),
            ("Switches of Play", "switches_of_play"),
            ("Shots", "shots"),
            ("Shots on Target", "shots_on_target"),
            ("Completed Dribbles", "dribbles_completed"),
            ("Tackles", "tackles"),
            ("Tackle Success Rate", "tackle_success_rate"),
            ("Ground Duels", "ground_duels_total"),
            ("Interceptions", "interceptions"),
            ("Blocks", "blocks"),
            ("Pressures", "pressures"),
            ("Touches", "touches"),
            ("Carries", "carries"),
            ("Progressive Carries", "progressive_carries"),
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
            console.print("[green]Radar chart generated.[/green]")
        except Exception as exc:
            console.print(f"[yellow]Warning: failed to generate radar chart: {exc}[/yellow]")

        # Similarity score
        similarity = result.get("similarity")
        if similarity is not None:
            cosine_dist = 1 - similarity
            console.print(
                Panel(
                    f"Cosine similarity: [bold]{similarity:.4f}[/bold]\n"
                    f"Cosine distance: [bold]{cosine_dist:.4f}[/bold]",
                    title="Player Similarity",
                    border_style="cyan",
                )
            )
        else:
            console.print("[yellow]Warning: embeddings not found to compute similarity.[/yellow]")

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error comparing players: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("find-similar")
def find_similar(
    player: str = typer.Argument(..., help="Reference player name"),
    season: Optional[str] = typer.Option(None, "--season", help="Filter by season"),
    limit: int = typer.Option(10, "--limit", help="Maximum number of results"),
) -> None:
    """Find similar players using pgvector vector search."""
    repo = _get_repo()

    try:
        use_case = FindSimilar(repo)
        result = use_case.execute(player, season, limit)

        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            raise typer.Exit(1)

        similar_df = result["similar_df"]
        resolved_season = result["season"]

        if resolved_season != season and season is None:
            console.print(f"[cyan]Using most recent season: {resolved_season}[/cyan]")

        if similar_df.empty:
            console.print(f"[yellow]No similar players found for '{player}'.[/yellow]")
            raise typer.Exit(0)

        table = Table(title=f"Players Similar to {player} ({resolved_season})")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Player", style="bold")
        table.add_column("Position")
        table.add_column("Distance", justify="right")
        table.add_column("Archetype")

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
        console.print(f"[red]Error finding similar players: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("recommend")
def recommend(
    profile_path: str = typer.Argument(..., help="Path to the JSON file with the desired profile"),
    season: Optional[str] = typer.Option(None, "--season", help="Filter by season"),
    limit: int = typer.Option(10, "--limit", help="Maximum number of recommendations"),
) -> None:
    """Recommend players based on a desired attribute profile."""
    try:
        profile_file = Path(profile_path)
        if not profile_file.exists():
            console.print(f"[red]Error: file '{profile_path}' not found.[/red]")
            raise typer.Exit(1)

        with open(profile_file, "r", encoding="utf-8") as f:
            profile = json.load(f)

        console.print(f"[cyan]Profile loaded: {len(profile)} attributes defined.[/cyan]")

    except json.JSONDecodeError as exc:
        console.print(f"[red]Error: invalid JSON file: {exc}[/red]")
        raise typer.Exit(1)

    repo = _get_repo()

    try:
        from football_moneyball.domain import embeddings as emb_mod

        with console.status("[bold green]Searching for recommendations..."):
            # Build synthetic embedding from profile using PCA/scaler
            embedding = emb_mod.profile_to_embedding(profile)
            recommendations = repo.recommend_by_profile(
                embedding=embedding,
                season=season or "",
                limit=limit,
                position_group=profile.get("position_group"),
            )

        if recommendations.empty:
            console.print("[yellow]No recommendations found for the given profile.[/yellow]")
            raise typer.Exit(0)

        table = Table(title="Player Recommendations")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Player", style="bold")
        table.add_column("Season")
        table.add_column("Score", justify="right")
        table.add_column("Archetype")

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

            with console.status("[bold green]Searching for recommendations..."):
                recommendations = player_embeddings.recommend_by_profile(
                    repo.session, profile, season=season, limit=limit
                )

            if recommendations.empty:
                console.print("[yellow]No recommendations found for the given profile.[/yellow]")
                raise typer.Exit(0)

            table = Table(title="Player Recommendations")
            table.add_column("#", justify="right", style="dim")
            table.add_column("Player", style="bold")
            table.add_column("Season")
            table.add_column("Score", justify="right")
            table.add_column("Archetype")

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
            console.print("[red]Error: embeddings module not available.[/red]")
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error generating recommendations: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("scout-report")
def scout_report(
    player: str = typer.Argument(..., help="Player name"),
    season: Optional[str] = typer.Option(None, "--season", help="Filter by season"),
    team_target: Optional[str] = typer.Option(None, "--team-target", help="Target team for fit analysis"),
    output: Optional[str] = typer.Option(None, "--output", help="Output file path (markdown or JSON)"),
) -> None:
    """Generate a complete scouting report for a player."""
    repo = _get_repo()

    try:
        use_case = GenerateReport(repo)
        report = use_case.execute(player, season, team_target)

        if "error" in report:
            console.print(f"[red]Error: {report['error']}[/red]")
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
                        f"# Scouting Report: {player}",
                        f"\n**Season:** {report['temporada']}",
                        f"**Matches:** {report['partidas']}",
                        f"**Archetype:** {report['arquetipo']}",
                        "\n## Total Metrics",
                    ]
                    for k, v in report["metricas_totais"].items():
                        lines.append(f"- {k}: {v}")
                    lines.append("\n## Per-90 Metrics")
                    for k, v in report["metricas_por_90"].items():
                        lines.append(f"- {k}: {v}")
                    if percentiles:
                        lines.append("\n## Percentiles")
                        for k, v in percentiles.items():
                            lines.append(f"- {k}: {v}%")
                    if report["similares"]:
                        lines.append("\n## Similar Players")
                        for s in report["similares"]:
                            lines.append(f"- {s.get('player_name', '')} (distance: {s.get('distance', '')})")

                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")

            console.print(f"[green]Report saved to: {output_path}[/green]")
        else:
            # Display in terminal
            header_text = (
                f"[bold]{player}[/bold]\n"
                f"Season: {report['temporada']}\n"
                f"Matches: {report['partidas']}\n"
                f"Archetype: [bold cyan]{report['arquetipo']}[/bold cyan]"
            )
            console.print(Panel(header_text, title="Scouting Report", border_style="blue"))

            # Metrics table
            metrics_table = Table(title="Metrics")
            metrics_table.add_column("Metric", style="bold")
            metrics_table.add_column("Total", justify="right")
            metrics_table.add_column("Per 90 min", justify="right")
            metrics_table.add_column("Percentile", justify="right")

            display_keys = [
                ("Goals", "goals"),
                ("Assists", "assists"),
                ("xG", "xg"),
                ("xA", "xa"),
                ("Big Chances", "big_chances"),
                ("Passes", "passes"),
                ("Completed Passes", "passes_completed"),
                ("Short Passes", "passes_short"),
                ("Long Passes", "passes_long"),
                ("Passes Under Pressure", "passes_under_pressure"),
                ("Progressive Passes", "progressive_passes"),
                ("Progressive Receptions", "progressive_receptions"),
                ("Key Passes", "key_passes"),
                ("Switches of Play", "switches_of_play"),
                ("Shots", "shots"),
                ("Completed Dribbles", "dribbles_completed"),
                ("Tackles", "tackles"),
                ("Tackle Success Rate %", "tackle_success_rate"),
                ("Ground Duels", "ground_duels_total"),
                ("Interceptions", "interceptions"),
                ("Pressures", "pressures"),
                ("Progressive Carries", "progressive_carries"),
                ("xT Generated", "xt_generated"),
                ("VAEP Generated", "vaep_generated"),
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
                sim_table = Table(title="Top 5 Similar Players")
                sim_table.add_column("Player", style="bold")
                sim_table.add_column("Distance", justify="right")
                sim_table.add_column("Archetype")

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
                        title=f"Fit with {team_target}",
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
        console.print(f"[red]Error generating scouting report: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("list-competitions")
def list_competitions() -> None:
    """List competitions available in StatsBomb open data."""
    try:
        provider = _get_provider()

        with console.status("[bold green]Fetching competitions..."):
            comps = provider.get_competitions()

        if comps.empty:
            console.print("[yellow]No competitions found.[/yellow]")
            raise typer.Exit(0)

        table = Table(title="Available Competitions (StatsBomb Open Data)")
        table.add_column("Comp. ID", justify="right", style="dim")
        table.add_column("Competition", style="bold")
        table.add_column("Season ID", justify="right", style="dim")
        table.add_column("Season")
        table.add_column("Country")

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
        console.print(f"[red]Error listing competitions: {exc}[/red]")
        raise typer.Exit(1)


@app.command("list-matches")
def list_matches(
    competition_id: int = typer.Argument(..., help="StatsBomb competition ID"),
    season_id: int = typer.Argument(..., help="StatsBomb season ID"),
) -> None:
    """List matches for a competition/season."""
    try:
        provider = _get_provider()

        with console.status("[bold green]Fetching matches..."):
            matches = provider.get_matches(competition_id, season_id)

        if matches.empty:
            console.print("[yellow]No matches found.[/yellow]")
            raise typer.Exit(0)

        table = Table(title=f"Matches - Competition {competition_id} / Season {season_id}")
        table.add_column("ID", justify="right", style="dim")
        table.add_column("Date")
        table.add_column("Home", style="bold")
        table.add_column("Score", justify="center")
        table.add_column("Away", style="bold")

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
        console.print(f"[red]Error listing matches: {exc}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# v0.4.0 — Betting commands
# ---------------------------------------------------------------------------

@app.command("predict")
def predict(
    match_id: int = typer.Argument(..., help="Match ID"),
    home_team: str = typer.Option(..., "--home", help="Home team name"),
    away_team: str = typer.Option(..., "--away", help="Away team name"),
    simulations: int = typer.Option(10_000, "--sims", help="Number of Monte Carlo simulations"),
) -> None:
    """Predict a match outcome via Monte Carlo + Poisson."""
    from football_moneyball.use_cases.predict_match import PredictMatch

    repo = get_repository()
    try:
        with console.status("[bold green]Running Monte Carlo simulation..."):
            result = PredictMatch(repo).execute(match_id, home_team, away_team, simulations)

        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            raise typer.Exit(1)

        # Header
        console.print(Panel(
            f"[bold]{home_team}[/bold] vs [bold]{away_team}[/bold]\n"
            f"Expected xG: {result['home_xg']:.2f} - {result['away_xg']:.2f}\n"
            f"Simulations: {result['simulations']:,}",
            title="Monte Carlo Prediction", border_style="cyan",
        ))

        # 1X2 probabilities
        prob_table = Table(title="1X2 Probabilities")
        prob_table.add_column("Outcome", style="bold")
        prob_table.add_column("Probability", justify="right")
        prob_table.add_column("Fair odds", justify="right")
        for label, key in [("Home", "home_win_prob"), ("Draw", "draw_prob"), ("Away", "away_win_prob")]:
            prob = result[key]
            fair_odds = round(1 / prob, 2) if prob > 0 else 0
            prob_table.add_row(label, f"{prob*100:.1f}%", f"{fair_odds:.2f}")
        console.print(prob_table)

        # Markets
        market_table = Table(title="Other Markets")
        market_table.add_column("Market", style="bold")
        market_table.add_column("Probability", justify="right")
        for label, key in [
            ("Over 0.5 goals", "over_05"), ("Over 1.5 goals", "over_15"),
            ("Over 2.5 goals", "over_25"), ("Over 3.5 goals", "over_35"),
            ("Both teams to score", "btts_prob"),
        ]:
            prob = result.get(key, 0)
            market_table.add_row(label, f"{prob*100:.1f}%")
        console.print(market_table)

        # Most likely scores
        console.print(f"\n[cyan]Most likely score:[/cyan] [bold]{result['most_likely_score']}[/bold]")
        if result.get("score_matrix"):
            score_table = Table(title="Top 5 Scores")
            score_table.add_column("Score", style="bold")
            score_table.add_column("Probability", justify="right")
            for score, prob in list(result["score_matrix"].items())[:5]:
                score_table.add_row(score, f"{prob*100:.1f}%")
            console.print(score_table)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Prediction error: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("value-bets")
def value_bets(
    bankroll: float = typer.Option(1000.0, "--bankroll", help="Current bankroll"),
    min_edge: float = typer.Option(0.05, "--min-edge", help="Minimum edge (0.05 = 5%)"),
    bookmaker: str = typer.Option("betfair", "--bookmaker", help="Bookmaker filter (default: betfair)"),
) -> None:
    """Find value bets on upcoming matches via Betfair Exchange."""
    from football_moneyball.config import get_odds_provider
    from football_moneyball.use_cases.find_value_bets import FindValueBets

    repo = get_repository()
    try:
        odds_provider = get_odds_provider()
        with console.status(f"[bold green]Fetching odds ({bookmaker})..."):
            result = FindValueBets(odds_provider, repo).execute(
                bankroll=bankroll, min_edge=min_edge, bookmaker_filter=bookmaker,
            )

        if not result.get("value_bets"):
            console.print(f"[yellow]No value bets found (edge > {min_edge*100:.0f}%).[/yellow]")
            console.print(f"Matches analyzed: {result.get('total_matches', 0)}")
            raise typer.Exit(0)

        console.print(Panel(
            f"Matches analyzed: {result['total_matches']}\n"
            f"Matches with value: {result['matches_with_value']}\n"
            f"Value bets found: {len(result['value_bets'])}",
            title="Value Bets — Betfair Exchange", border_style="green",
        ))

        vb_table = Table(title="Recommended Value Bets")
        vb_table.add_column("Match", style="bold")
        vb_table.add_column("Market")
        vb_table.add_column("Bet")
        vb_table.add_column("Model", justify="right")
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
        console.print(f"[red]Error finding value bets: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("backtest")
def backtest(
    season: str = typer.Option("2026", "--season", help="Season"),
    competition: str = typer.Option("Brasileirão Série A", "--competition", help="Competition"),
    bankroll: float = typer.Option(1000.0, "--bankroll", help="Initial bankroll"),
    min_edge: float = typer.Option(0.05, "--min-edge", help="Minimum edge (0.05 = 5%)"),
) -> None:
    """Run backtesting using historical Brasileirão data."""
    from football_moneyball.use_cases.backtest import Backtest

    repo = get_repository()
    try:
        with console.status("[bold green]Running backtesting..."):
            result = Backtest(repo).execute(
                competition=competition,
                season=season,
                initial_bankroll=bankroll,
                min_edge=min_edge,
            )

        if "error" in result:
            console.print(f"[yellow]{result['error']}[/yellow]")
            if "predictions" in result:
                console.print(f"Matches analyzed: {result.get('matches_analyzed', 0)}")
            raise typer.Exit(0)

        # Results panel
        roi_color = "green" if result["roi"] > 0 else "red"
        console.print(Panel(
            f"[bold]Backtesting — {competition} {season}[/bold]\n\n"
            f"Initial bankroll: R$ {result['initial_bankroll']:.2f}\n"
            f"Final bankroll: R$ {result['final_bankroll']:.2f}\n"
            f"[{roi_color}]ROI: {result['roi']:+.2f}%[/{roi_color}]\n\n"
            f"Matches analyzed: {result['matches_analyzed']}\n"
            f"Bets placed: {result['bets_placed']}\n"
            f"Bets won: {result['bets_won']}\n"
            f"Hit rate: {result['hit_rate']:.1f}%\n"
            f"Average edge: {result['avg_edge']:.2f}%\n"
            f"Average odds: {result['avg_odds']:.2f}\n\n"
            f"Brier score: {result['brier_score']:.4f} (< 0.25 = better than random)\n"
            f"Max drawdown: {result['max_drawdown']:.1f}%\n"
            f"Total staked: R$ {result['total_staked']:.2f}\n"
            f"Total return: R$ {result['total_return']:.2f}",
            title="Backtesting Results", border_style=roi_color,
        ))

        # Top bets
        if result.get("bets"):
            bets_table = Table(title="Last 10 Bets")
            bets_table.add_column("Match", style="bold")
            bets_table.add_column("Market")
            bets_table.add_column("Bet")
            bets_table.add_column("Odds", justify="right")
            bets_table.add_column("Edge", justify="right")
            bets_table.add_column("Stake", justify="right")
            bets_table.add_column("Result")
            bets_table.add_column("Profit", justify="right")

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
        console.print(f"[red]Backtesting error: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("verify")
def verify(
    season: str = typer.Option("2026", "--season", help="Season"),
    competition: str = typer.Option("Brasileirão Série A", "--competition", help="Competition"),
) -> None:
    """Verify predictions against actual results."""
    from football_moneyball.use_cases.verify_predictions import VerifyPredictions

    repo = get_repository()
    try:
        with console.status("[bold green]Comparing predictions with results..."):
            result = VerifyPredictions(repo).execute(competition=competition, season=season)

        if "error" in result:
            console.print(f"[yellow]{result['error']}[/yellow]")
            raise typer.Exit(0)

        # Summary
        acc_color = "green" if result["accuracy_1x2"] > 40 else "red"
        brier_color = "green" if result["avg_brier_score"] < 0.25 else "yellow"
        console.print(Panel(
            f"[bold]Verification — {competition} {season}[/bold]\n\n"
            f"Matches verified: {result['total_matches']}\n\n"
            f"[{acc_color}]Correct 1X2: {result['correct_1x2']}/{result['total_matches']} "
            f"({result['accuracy_1x2']}%)[/{acc_color}]\n"
            f"Correct Over/Under: {result['correct_over_under']}/{result['total_matches']} "
            f"({result['accuracy_over_under']}%)\n"
            f"[{brier_color}]Brier score: {result['avg_brier_score']:.4f}[/{brier_color}] "
            f"(< 0.25 = better than random)",
            title="Model vs Reality", border_style=acc_color,
        ))

        # Detail table
        detail_table = Table(title="Per-Match Breakdown")
        detail_table.add_column("Match", style="bold")
        detail_table.add_column("Score")
        detail_table.add_column("Pred 1X2")
        detail_table.add_column("Actual")
        detail_table.add_column("Correct")
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
        console.print(f"[red]Verification error: {exc}[/red]")
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
    competition_id: int = typer.Option(325, "--competition-id", help="Sofascore tournament_id"),
    season_id: int = typer.Option(87678, "--season-id", help="Sofascore season_id (2026=87678, 2025=72034, 2024=58766)"),
) -> None:
    """Ingest new matches from the provider (delta — only missing games)."""
    from football_moneyball.use_cases.ingest_matches import IngestMatches

    repo = get_repository()
    try:
        if provider_name == "sofascore":
            from football_moneyball.adapters.sofascore_provider import SofascoreProvider
            provider = SofascoreProvider(
                tournament_id=competition_id, season_id=season_id,
                competition_name=competition, season_name=season,
            )
        else:
            provider = get_provider(provider_name)
        with console.status(f"[bold green]Ingesting {season} from {provider_name}..."):
            result = IngestMatches(provider, repo).execute(
                competition=competition, season=season,
                competition_id=competition_id, season_id=season_id,
            )

        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            raise typer.Exit(1)

        console.print(Panel(
            f"Ingested: {result['ingested']}\n"
            f"Already existed: {result['skipped']}\n"
            f"Errors: {result['errors']}\n"
            f"Total in provider: {result['total']}",
            title=f"Ingestion — {provider_name}", border_style="green",
        ))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Ingestion error: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("train-models")
def train_models_cmd(
    season: str = typer.Option("all", "--season", help="Season or 'all' for all seasons"),
    models_dir: str = typer.Option(
        "football_moneyball/models", "--models-dir",
    ),
) -> None:
    """Train ML models (GBR) for goals, corners, cards."""
    from football_moneyball.use_cases.train_ml_models import TrainMLModels
    season_arg: str | None = None if season == "all" else season

    repo = get_repository()
    try:
        with console.status("[bold green]Training models..."):
            result = TrainMLModels(repo, models_dir).execute(season_arg)

        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            raise typer.Exit(1)

        table = Table(title="Trained ML Models", border_style="green")
        table.add_column("Target")
        table.add_column("MAE (CV)")
        table.add_column("Samples")
        table.add_column("File")
        for target, metrics in result.items():
            if "error" in metrics:
                table.add_row(target, "—", "—", f"[red]{metrics['error']}[/red]")
            else:
                table.add_row(
                    target,
                    f"{metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f}",
                    str(metrics["n_samples"]),
                    metrics.get("saved_to", ""),
                )
        console.print(table)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("train-catboost")
def train_catboost_cmd(
    models_dir: str = typer.Option("football_moneyball/models", "--models-dir"),
    seasons: str = typer.Option("2022,2023,2024,2025,2026", "--seasons"),
    draw_weight: float = typer.Option(1.3, "--draw-weight", help="Extra weight for draws"),
) -> None:
    """Train CatBoost 1x2 with Pi-Rating + EMA form + xG."""
    from football_moneyball.use_cases.train_catboost import TrainCatBoost

    repo = get_repository()
    try:
        seasons_list = [s.strip() for s in seasons.split(",")]
        with console.status("[bold green]Training CatBoost 1x2..."):
            result = TrainCatBoost(repo, models_dir).execute(
                seasons=seasons_list, draw_weight=draw_weight,
            )

        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            raise typer.Exit(1)

        fi = result.get("feature_importance", {})
        fi_sorted = sorted(fi.items(), key=lambda x: -x[1])[:8]
        fi_text = "\n".join(f"  {k:30s} {v:.1f}" for k, v in fi_sorted)

        console.print(Panel(
            f"[bold]CatBoost 1x2 trained[/bold]\n\n"
            f"Samples: {result['n_samples']} (of {result['total_matches']} matches)\n"
            f"Seasons: {result['per_season']}\n"
            f"Draw weight: {result['draw_weight']}\n"
            f"Best iteration: {result.get('best_iteration', '?')}\n\n"
            f"[bold]Validation metrics:[/bold]\n"
            f"RPS:      {result['rps']:.4f}\n"
            f"Accuracy: {result['accuracy']:.1f}%\n"
            f"Brier:    {result['brier']:.4f}\n\n"
            f"[bold]Pred distribution:[/bold] {result['pred_distribution']}\n\n"
            f"[bold]Feature importance (top 8):[/bold]\n{fi_text}\n\n"
            f"Saved to: {result['saved_to']}",
            title="CatBoost 1x2", border_style="green",
        ))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("fit-calibration")
def fit_calibration_cmd(
    models_dir: str = typer.Option("football_moneyball/models", "--models-dir"),
    seasons: str = typer.Option("2024,2026", "--seasons", help="Comma-separated seasons"),
    method: str = typer.Option("auto", "--method", help="Method: auto|platt|isotonic|temperature"),
) -> None:
    """Fit calibration (Dixon-Coles rho + Platt/Isotonic/Temperature) on historical data."""
    from football_moneyball.use_cases.fit_calibration import FitCalibration

    repo = get_repository()
    try:
        seasons_list = [s.strip() for s in seasons.split(",")]
        with console.status("[bold green]Fitting calibration..."):
            result = FitCalibration(repo, models_dir).execute(
                seasons=seasons_list, method=method,
            )

        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            raise typer.Exit(1)

        cv = result["cv_results"]
        cv_table = (
            f"[bold]CV comparison (val split):[/bold]\n"
            f"  platt:       brier={cv['platt']['brier_val']:.4f}  ece={cv['platt']['ece_val']:.4f}\n"
            f"  isotonic:    brier={cv['isotonic']['brier_val']:.4f}  ece={cv['isotonic']['ece_val']:.4f}\n"
            f"  temperature: brier={cv['temperature']['brier_val']:.4f}  ece={cv['temperature']['ece_val']:.4f}\n"
        )

        console.print(Panel(
            f"[bold]Calibration fitted[/bold]\n\n"
            f"Calibration method: [bold green]{result['method']}[/bold green]\n"
            f"Score method: {result.get('score_method', 'dixon-coles')}\n"
            f"Dixon-Coles rho: {result['rho']:.4f} | "
            f"Bivariate lambda3: {result.get('lambda3', 0):.4f}\n"
            f"Samples: {result['n_samples']} (train={result['n_train']}, val={result['n_val']})\n"
            f"Per season: {result['per_season']}\n\n"
            f"{cv_table}\n"
            f"[bold]Full-sample metrics ({result['method']}):[/bold]\n"
            f"Brier raw → cal:    {result['brier_raw']:.4f} → {result['brier_calibrated']:.4f}\n"
            f"ECE raw → cal:      {result['ece_raw']:.4f} → {result['ece_calibrated']:.4f}\n"
            f"Accuracy raw → cal: {result['accuracy_raw']:.1f}% → {result['accuracy_calibrated']:.1f}%\n\n"
            f"Saved to: {result['saved_to']}",
            title="Calibration", border_style="green",
        ))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("ingest-context")
def ingest_context_cmd(
    season: str = typer.Option("2026", "--season"),
    backfill: bool = typer.Option(False, "--backfill", help="Process ALL matches"),
    provider_name: str = typer.Option("sofascore", "--provider"),
) -> None:
    """Ingest managers, injuries and standings (context features v1.6.0)."""
    from football_moneyball.use_cases.ingest_context import IngestContext

    repo = get_repository()
    try:
        provider = get_provider(provider_name)
        with console.status("[bold green]Ingesting context..."):
            result = IngestContext(provider, repo).execute(season=season, backfill=backfill)

        console.print(Panel(
            f"Matches processed: {result['matches_processed']}\n"
            f"Managers found: {result['managers_found']}\n"
            f"Injuries saved: {result['injuries_saved']}\n"
            f"Coaches persisted: {result['coaches_persisted']}\n"
            f"Standings saved: {result['standings_saved']}\n"
            f"Errors: {result['errors']}",
            title="Ingest Context", border_style="green",
        ))
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("ingest-lineups")
def ingest_lineups_cmd(
    match_ids: str = typer.Option("", "--match-ids", help="Comma-separated IDs"),
    provider_name: str = typer.Option("sofascore", "--provider"),
) -> None:
    """Ingest confirmed lineups for matches."""
    from football_moneyball.use_cases.ingest_lineups import IngestLineups

    if not match_ids:
        console.print("[red]Provide --match-ids with comma-separated IDs[/red]")
        raise typer.Exit(1)

    ids = [int(x) for x in match_ids.split(",") if x.strip()]
    repo = get_repository()
    try:
        provider = get_provider(provider_name)
        with console.status("[bold green]Ingesting lineups..."):
            result = IngestLineups(provider, repo).execute(match_ids=ids)

        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            raise typer.Exit(1)

        console.print(Panel(
            f"Lineups ingested: {result['ingested']}\n"
            f"Errors: {result['errors']}",
            title="Ingest Lineups", border_style="green",
        ))
        for d in result.get("details", []):
            console.print(f"  {d['home']} vs {d['away']}: {d['players']} players")
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("snapshot-odds")
def snapshot_odds() -> None:
    """Fetch current odds and save to PostgreSQL."""
    from football_moneyball.use_cases.snapshot_odds import SnapshotOdds
    from football_moneyball.config import get_odds_provider

    repo = get_repository()
    try:
        odds_provider = get_odds_provider()
        odds_provider.repo = repo
        with console.status("[bold green]Fetching odds..."):
            result = SnapshotOdds(odds_provider, repo).execute()

        if "error" in result:
            console.print(f"[yellow]{result['error']}[/yellow]")
            raise typer.Exit(0)

        console.print(Panel(
            f"Matches: {result['matches']}\n"
            f"Bookmakers: {result['bookmakers']}\n"
            f"Total odds: {result['total_odds']}",
            title="Odds Snapshot", border_style="green",
        ))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error fetching odds: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("predict-all")
def predict_all_cmd(
    competition: str = typer.Option("Brasileirão Série A", "--competition"),
    season: str = typer.Option("2026", "--season"),
) -> None:
    """Predict all pending matches of the matchday."""
    from football_moneyball.use_cases.predict_all import PredictAll

    repo = get_repository()
    try:
        with console.status("[bold green]Predicting matches..."):
            result = PredictAll(repo).execute(competition, season)

        if "error" in result:
            console.print(f"[yellow]{result['error']}[/yellow]")
            raise typer.Exit(0)

        preds = result.get("predictions", [])
        if not preds:
            console.print("[yellow]No predictions generated.[/yellow]")
            raise typer.Exit(0)

        table = Table(title=f"Predictions — {result['total']} matches")
        table.add_column("Match", style="bold")
        table.add_column("xG H", justify="right")
        table.add_column("xG A", justify="right")
        table.add_column("P(H)", justify="right")
        table.add_column("P(D)", justify="right")
        table.add_column("P(A)", justify="right")
        table.add_column("O/U 2.5", justify="right")
        table.add_column("Score", justify="center")

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
        console.print(f"[red]Prediction error: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# v0.9.0 — Track Record commands
# ---------------------------------------------------------------------------

@app.command("resolve")
def resolve_cmd() -> None:
    """Resolve pending predictions with actual results."""
    from football_moneyball.use_cases.resolve_predictions import ResolvePredictions

    repo = get_repository()
    try:
        with console.status("[bold green]Resolving predictions..."):
            result = ResolvePredictions(repo).execute()

        console.print(Panel(
            f"Resolved: {result['resolved']}\n"
            f"Pending: {result['still_pending']}\n"
            f"Errors: {result.get('errors', 0)}",
            title="Resolve", border_style="green",
        ))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error resolving predictions: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


@app.command("track-record")
def track_record_cmd() -> None:
    """Show the model's track record."""
    from football_moneyball.domain.track_record import calculate_track_record

    repo = get_repository()
    try:
        preds = repo.get_prediction_history()
        tr = calculate_track_record(preds)

        # Summary panel
        if tr["resolved"] == 0:
            console.print(Panel(
                f"Total: {tr['total']} predictions ({tr['pending']} pending)\n"
                f"No predictions resolved yet. Run [bold]moneyball resolve[/bold] first.",
                title="Track Record", border_style="yellow",
            ))
            return

        acc_color = "green" if tr["accuracy_1x2"] > 40 else "red"
        brier_color = "green" if tr["avg_brier"] < 0.25 else "yellow"
        console.print(Panel(
            f"Total: {tr['total']} predictions ({tr['resolved']} resolved, {tr['pending']} pending)\n"
            f"[{acc_color}]1X2 accuracy: {tr['accuracy_1x2']:.1f}%[/{acc_color}]\n"
            f"O/U accuracy: {tr['accuracy_over_under']:.1f}%\n"
            f"[{brier_color}]Brier: {tr['avg_brier']:.4f}[/{brier_color}]",
            title="Track Record", border_style="cyan",
        ))

        # By round table
        if tr["by_round"]:
            round_table = Table(title="Accuracy by Matchday")
            round_table.add_column("Matchday", justify="right")
            round_table.add_column("Games", justify="right")
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

            team_table = Table(title="Accuracy by Team (Top 10)")
            team_table.add_column("Team", style="bold")
            team_table.add_column("Games", justify="right")
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
        console.print(f"[red]Error displaying track record: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
