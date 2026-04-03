"""Adapter Matplotlib — implementa Visualizer via matplotlib + mplsoccer.

Gera graficos e visualizacoes de redes de passes, radares de comparacao
de jogadores, mapas de calor de acoes, grafos de sinergia e rankings RAPM.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mplsoccer import Pitch, Radar


class MatplotlibVisualizer:
    """Implementacao do port Visualizer usando matplotlib e mplsoccer."""

    @staticmethod
    def _save_and_return(fig: Figure, save_path: str | None) -> Figure:
        """Aplica tight_layout, salva e retorna a figura."""
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig

    def plot_pass_network(
        self,
        G: nx.DiGraph,
        team: str,
        match_info: dict | None = None,
        save_path: str | None = None,
    ) -> Figure:
        """Desenha rede de passes sobre campo StatsBomb."""
        plt.style.use("dark_background")
        pitch = Pitch(pitch_type="statsbomb", line_color="white", pitch_color="#1a1a2e")
        fig, ax = pitch.draw(figsize=(12, 8))

        if len(G.nodes) == 0:
            ax.set_title(f"Rede de Passes - {team} (sem dados)", color="white", fontsize=16)
            return self._save_and_return(fig, save_path)

        node_positions = {}
        for node in G.nodes:
            nd = G.nodes[node]
            node_positions[node] = (nd.get("avg_x", 60), nd.get("avg_y", 40))

        degree_centrality = nx.degree_centrality(G)
        max_centrality = max(degree_centrality.values()) if degree_centrality else 1
        min_size, max_size = 200, 1500

        edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges]
        max_weight = max(edge_weights) if edge_weights else 1

        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 1)
            x_s, y_s = node_positions[u]
            x_e, y_e = node_positions[v]
            nw = weight / max_weight
            pitch.arrows(x_s, y_s, x_e, y_e, ax=ax, width=0.5 + nw * 4.5,
                         headwidth=4, headlength=4, color="#00d4ff", alpha=0.2 + nw * 0.6, zorder=2)

        for node in G.nodes:
            x, y = node_positions[node]
            c = degree_centrality.get(node, 0)
            size = min_size + (c / max_centrality) * (max_size - min_size)
            ax.scatter(x, y, s=size, c="#e94560", edgecolors="white", linewidths=1.5, zorder=3)
            raw = G.nodes[node].get("player_name", str(node))
            label = raw if len(str(raw)) <= 15 else str(raw)[:12] + "..."
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7, color="white", fontweight="bold", zorder=4)

        title = f"Rede de Passes - {team}"
        if match_info:
            parts = []
            if "opponent" in match_info:
                parts.append(f"vs {match_info['opponent']}")
            if "date" in match_info:
                parts.append(str(match_info["date"]))
            if "score" in match_info:
                parts.append(match_info["score"])
            if parts:
                title += f"\n{' | '.join(parts)}"
        ax.set_title(title, color="white", fontsize=16, fontweight="bold", pad=10)
        return self._save_and_return(fig, save_path)

    def plot_radar_comparison(
        self, player_a: dict, player_b: dict,
        metrics: list[str] | None = None, save_path: str | None = None,
    ) -> Figure:
        """Radar chart comparando dois jogadores."""
        plt.style.use("dark_background")
        if metrics is None:
            metrics = ["goals", "assists", "xg", "key_passes", "progressive_passes",
                       "tackles", "interceptions", "pressures", "dribbles_completed", "pass_pct"]

        low = [0] * len(metrics)
        high = [100] * len(metrics)
        display_labels = [m.replace("_", " ").title() for m in metrics]

        radar = Radar(display_labels, low, high, num_rings=5, ring_width=1, center_circle_radius=1)
        values_a = [player_a.get(m, 0) for m in metrics]
        values_b = [player_b.get(m, 0) for m in metrics]
        name_a = player_a.get("name", "Jogador A")
        name_b = player_b.get("name", "Jogador B")

        fig, ax = radar.setup_axis()
        fig.set_facecolor("#1a1a2e")
        fig.set_size_inches(10, 10)
        radar.draw_circles(ax=ax, facecolor="#16213e", edgecolor="#0f3460")
        radar_output = radar.draw_radar_compare(
            values_a, values_b, ax=ax,
            kwargs_radar={"facecolor": "#e94560", "alpha": 0.55},
            kwargs_compare={"facecolor": "#00d4ff", "alpha": 0.55},
        )
        radar_poly, radar_poly2, _, _ = radar_output
        radar.draw_range_labels(ax=ax, fontsize=7, color="white", alpha=0.5)
        radar.draw_param_labels(ax=ax, fontsize=10, color="white", fontweight="bold")
        ax.legend([radar_poly, radar_poly2], [name_a, name_b], loc="upper right",
                  fontsize=12, facecolor="#16213e", edgecolor="white", labelcolor="white")
        fig.suptitle(f"Comparacao: {name_a} vs {name_b}", color="white", fontsize=18, fontweight="bold", y=0.97)
        return self._save_and_return(fig, save_path)

    def plot_action_heatmap(
        self, events_df: pd.DataFrame, player_name: str,
        action_type: str | None = None, save_path: str | None = None,
    ) -> Figure:
        """Mapa de calor das acoes de um jogador."""
        plt.style.use("dark_background")
        mask = events_df["player"] == player_name
        if action_type:
            mask = mask & (events_df["type"] == action_type)
        filtered = events_df.loc[mask].copy()

        if "location" in filtered.columns:
            valid = filtered["location"].dropna()
            x_coords = valid.apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) else np.nan)
            y_coords = valid.apply(lambda loc: loc[1] if isinstance(loc, (list, tuple)) else np.nan)
        elif "x" in filtered.columns and "y" in filtered.columns:
            x_coords, y_coords = filtered["x"], filtered["y"]
        else:
            x_coords = pd.Series(dtype=float)
            y_coords = pd.Series(dtype=float)

        valid_mask = x_coords.notna() & y_coords.notna()
        x_coords = x_coords[valid_mask].astype(float).values
        y_coords = y_coords[valid_mask].astype(float).values

        pitch = Pitch(pitch_type="statsbomb", line_color="white", pitch_color="#1a1a2e")
        fig, ax = pitch.draw(figsize=(12, 8))

        if len(x_coords) > 0:
            bin_stat = pitch.bin_statistic(x_coords, y_coords, statistic="count", bins=(25, 25))
            bin_stat["statistic"] = np.where(bin_stat["statistic"] == 0, np.nan, bin_stat["statistic"])
            pitch.heatmap(bin_stat, ax=ax, cmap="hot", edgecolors="#1a1a2e", zorder=1)
            pitch.scatter(x_coords, y_coords, ax=ax, s=15, c="white", alpha=0.3, zorder=2)

        action_label = f" ({action_type})" if action_type else ""
        ax.set_title(f"Mapa de Calor - {player_name}{action_label}",
                     color="white", fontsize=16, fontweight="bold", pad=10)
        return self._save_and_return(fig, save_path)

    def plot_synergy_graph(
        self, compatibility_df: pd.DataFrame,
        team: str | None = None, save_path: str | None = None,
    ) -> Figure:
        """Grafo de sinergia/compatibilidade entre jogadores."""
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        G = nx.Graph()

        if compatibility_df.empty:
            title = "Sinergia entre Jogadores" + (f" - {team}" if team else "")
            ax.set_title(title, color="white", fontsize=16)
            ax.text(0.5, 0.5, "Sem dados", ha="center", va="center", color="white", fontsize=14, transform=ax.transAxes)
            ax.axis("off")
            return self._save_and_return(fig, save_path)

        threshold = compatibility_df["score"].quantile(0.5)
        strong = compatibility_df[compatibility_df["score"] >= threshold]
        for _, row in strong.iterrows():
            G.add_edge(row["player_a"], row["player_b"], weight=row["score"])

        if len(G.nodes) == 0:
            ax.text(0.5, 0.5, "Sem conexoes acima do limiar", ha="center", va="center",
                    color="white", fontsize=14, transform=ax.transAxes)
            ax.axis("off")
            return self._save_and_return(fig, save_path)

        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
        degrees = dict(G.degree())
        node_colors = [degrees[n] for n in G.nodes]
        max_degree = max(node_colors) if node_colors else 1
        edge_weights = [G[u][v]["weight"] for u, v in G.edges]
        max_ew = max(edge_weights) if edge_weights else 1

        for (u, v), width, alpha in zip(G.edges, [1 + (w / max_ew) * 5 for w in edge_weights],
                                         [0.3 + (w / max_ew) * 0.5 for w in edge_weights]):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#00d4ff", linewidth=width, alpha=alpha, zorder=1)

        scatter = nx.draw_networkx_nodes(G, pos, ax=ax,
            node_size=[300 + (degrees[n] / max_degree) * 700 for n in G.nodes],
            node_color=node_colors, cmap=plt.cm.plasma, edgecolors="white", linewidths=1.5, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")

        title = "Sinergia entre Jogadores" + (f" - {team}" if team else "")
        ax.set_title(title, color="white", fontsize=16, fontweight="bold")
        ax.axis("off")
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Grau de Conexao", color="white", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        return self._save_and_return(fig, save_path)

    def plot_rapm_rankings(
        self, rapm_df: pd.DataFrame, top_n: int = 20, save_path: str | None = None,
    ) -> Figure:
        """Ranking RAPM em barras horizontais."""
        plt.style.use("dark_background")
        df = rapm_df.copy()
        if "rapm" not in df.columns and "rapm_value" in df.columns:
            df = df.rename(columns={"rapm_value": "rapm"})
        if "player" not in df.columns and "player_name" in df.columns:
            df = df.rename(columns={"player_name": "player"})
        df = df.sort_values("rapm", ascending=False)

        top = df.head(top_n)
        bottom = df.tail(top_n)
        display_df = pd.concat([top, bottom]).drop_duplicates().sort_values("rapm", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(8, len(display_df) * 0.35)))
        fig.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in display_df["rapm"]]
        bars = ax.barh(display_df["player"], display_df["rapm"], color=colors, edgecolor="white", linewidth=0.3, alpha=0.85)

        for bar, value in zip(bars, display_df["rapm"]):
            offset = 0.02 * (display_df["rapm"].max() - display_df["rapm"].min())
            ha = "left" if value >= 0 else "right"
            x_label = bar.get_width() + offset if value >= 0 else bar.get_width() - offset
            ax.text(x_label, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                    ha=ha, va="center", color="white", fontsize=8, fontweight="bold")

        ax.axvline(x=0, color="white", linewidth=0.8, alpha=0.5)
        ax.set_title("RAPM - Impacto Individual Ajustado", color="white", fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("RAPM", color="white", fontsize=12)
        ax.tick_params(colors="white", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        return self._save_and_return(fig, save_path)

    def plot_xt_heatmap(
        self, xt_grid: np.ndarray, l: int = 16, w: int = 12, save_path: str | None = None,
    ) -> Figure:
        """Superficie xT sobre o campo."""
        plt.style.use("dark_background")
        pitch = Pitch(pitch_type="statsbomb", line_color="white", line_alpha=0.3)
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.set_facecolor("#1a1a2e")
        ax.imshow(xt_grid.T, extent=[0, 120, 80, 0], cmap="YlOrRd", alpha=0.75, interpolation="bilinear", aspect="auto")
        ax.set_title("Expected Threat (xT) — Superficie de Ameaca", color="white", fontsize=16, fontweight="bold", pad=15)
        return self._save_and_return(fig, save_path)

    def plot_pressing_zones(
        self, pressing_data: dict, team: str, save_path: str | None = None,
    ) -> Figure:
        """Distribuicao de pressing por zona."""
        plt.style.use("dark_background")
        pitch = Pitch(pitch_type="statsbomb", line_color="white", line_alpha=0.3)
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.set_facecolor("#1a1a2e")

        zones = [pressing_data.get(f"pressing_zone_{i}", 0) for i in range(1, 7)]
        max_zone = max(zones) if max(zones) > 0 else 1
        boundaries = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120)]

        for (x0, x1), pct in zip(boundaries, zones):
            alpha = 0.2 + 0.6 * (pct / max_zone)
            ax.fill_between([x0, x1], 0, 80, color="#e74c3c", alpha=alpha)
            ax.text((x0 + x1) / 2, 40, f"{pct:.0f}%", ha="center", va="center",
                    color="white", fontsize=14, fontweight="bold")

        ppda = pressing_data.get("ppda", 0)
        success = pressing_data.get("pressing_success_rate", 0)
        cp_frac = pressing_data.get("counter_pressing_fraction", 0)
        info = f"PPDA: {ppda:.1f}  |  Sucesso: {success:.0f}%  |  Counter-press: {cp_frac:.0f}%"
        ax.set_title(f"Pressing Zones — {team}\n{info}", color="white", fontsize=14, fontweight="bold", pad=15)
        return self._save_and_return(fig, save_path)

    def plot_shot_map(
        self, shots_df: pd.DataFrame, player_name: str, save_path: str | None = None,
    ) -> Figure:
        """Mapa de chutes com xG e resultado."""
        plt.style.use("dark_background")
        pitch = Pitch(pitch_type="statsbomb", line_color="white", line_alpha=0.5)
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.set_facecolor("#1a1a2e")

        outcome_colors = {"Goal": "#2ecc71", "Saved": "#f39c12", "Saved To Post": "#f39c12",
                          "Blocked": "#95a5a6", "Off T": "#e74c3c", "Wayward": "#e74c3c", "Post": "#e67e22"}

        for _, row in shots_df.iterrows():
            loc = row.get("location")
            if not isinstance(loc, list) or len(loc) < 2:
                continue
            xg = row.get("shot_statsbomb_xg", 0.05) or 0.05
            outcome = row.get("shot_outcome", "Off T")
            color = outcome_colors.get(outcome, "#95a5a6")
            is_big = xg >= 0.3
            ax.scatter(loc[0], loc[1], s=xg * 800 + 30, c=color, alpha=0.85,
                       edgecolors="white" if is_big else "none", linewidths=2 if is_big else 0, zorder=3)

        for label, color in [("Gol", "#2ecc71"), ("Defesa", "#f39c12"), ("Bloqueado", "#95a5a6"), ("Fora", "#e74c3c")]:
            ax.scatter([], [], c=color, s=80, label=label)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.5)

        total_xg = shots_df.get("shot_statsbomb_xg", pd.Series([0])).sum()
        n_shots = len(shots_df)
        ax.set_title(f"Mapa de Chutes — {player_name}\n{n_shots} chutes | xG total: {total_xg:.2f}",
                     color="white", fontsize=14, fontweight="bold", pad=15)
        return self._save_and_return(fig, save_path)
