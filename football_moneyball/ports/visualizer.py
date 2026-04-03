"""Port para visualizacao de dados."""

from __future__ import annotations

from typing import Any, Protocol

import networkx as nx
import numpy as np
import pandas as pd


class Visualizer(Protocol):
    """Interface para geracao de visualizacoes.

    Define o contrato que qualquer implementacao de visualizacao deve seguir.
    A implementacao padrao utiliza matplotlib + mplsoccer, mas o sistema pode
    ser estendido para outros backends (plotly, bokeh, etc).

    Todas as funcoes retornam um objeto Figure (matplotlib ou equivalente).
    O parametro save_path, quando fornecido, salva a figura no caminho
    indicado.
    """

    def plot_pass_network(
        self,
        G: nx.DiGraph,
        team: str,
        match_info: dict | None = None,
        save_path: str | None = None,
    ) -> Any:
        """Desenha a rede de passes sobre um campo de futebol.

        Parameters
        ----------
        G : nx.DiGraph
            Grafo direcionado com atributos de no (avg_x, avg_y) e
            arestas ponderadas (weight). Coordenadas StatsBomb: x 0-120,
            y 0-80.
        team : str
            Nome do time para o titulo.
        match_info : dict, optional
            Metadados da partida (opponent, date, score) para o titulo.
        save_path : str, optional
            Caminho para salvar a figura.

        Returns
        -------
        Figure
            Objeto Figure com a visualizacao da rede de passes.
        """
        ...

    def plot_radar_comparison(
        self,
        player_a: dict,
        player_b: dict,
        metrics: list[str] | None = None,
        save_path: str | None = None,
    ) -> Any:
        """Radar chart comparando dois jogadores.

        Parameters
        ----------
        player_a : dict
            Dicionario com 'name' (str) e valores percentuais (0-100)
            para cada metrica.
        player_b : dict
            Dicionario com 'name' (str) e valores percentuais (0-100)
            para cada metrica.
        metrics : list[str], optional
            Lista de metricas a comparar. Se None, usa metricas padrao.
        save_path : str, optional
            Caminho para salvar a figura.

        Returns
        -------
        Figure
            Objeto Figure com o radar de comparacao.
        """
        ...

    def plot_action_heatmap(
        self,
        events_df: pd.DataFrame,
        player_name: str,
        action_type: str | None = None,
        save_path: str | None = None,
    ) -> Any:
        """Mapa de calor das acoes de um jogador no campo.

        Parameters
        ----------
        events_df : pd.DataFrame
            DataFrame de eventos com colunas 'player', 'type', 'location'
            (ou 'x', 'y'). Coordenadas StatsBomb.
        player_name : str
            Nome do jogador para filtrar.
        action_type : str, optional
            Tipo de acao para filtrar (ex: 'Pass', 'Shot', 'Carry').
        save_path : str, optional
            Caminho para salvar a figura.

        Returns
        -------
        Figure
            Objeto Figure com o mapa de calor.
        """
        ...

    def plot_xt_heatmap(
        self,
        xt_grid: np.ndarray,
        l: int = 16,
        w: int = 12,
        save_path: str | None = None,
    ) -> Any:
        """Desenha a superficie de Expected Threat (xT) sobre o campo.

        Parameters
        ----------
        xt_grid : np.ndarray
            Matriz (l, w) com valores xT por zona.
        l : int
            Celulas no eixo x.
        w : int
            Celulas no eixo y.
        save_path : str, optional
            Caminho para salvar a figura.

        Returns
        -------
        Figure
            Objeto Figure com o heatmap xT.
        """
        ...

    def plot_pressing_zones(
        self,
        pressing_data: dict,
        team: str,
        save_path: str | None = None,
    ) -> Any:
        """Visualiza a distribuicao de pressing por zona e metricas-chave.

        Parameters
        ----------
        pressing_data : dict
            Dicionario com chaves: ppda, pressing_success_rate,
            counter_pressing_fraction, pressing_zone_1..6.
        team : str
            Nome do time.
        save_path : str, optional
            Caminho para salvar.

        Returns
        -------
        Figure
            Objeto Figure com a visualizacao de zonas de pressing.
        """
        ...

    def plot_shot_map(
        self,
        shots_df: pd.DataFrame,
        player_name: str,
        save_path: str | None = None,
    ) -> Any:
        """Mapa de chutes com tamanho proporcional ao xG e cor por resultado.

        Parameters
        ----------
        shots_df : pd.DataFrame
            DataFrame de eventos de chute com colunas: location,
            shot_outcome, shot_statsbomb_xg.
        player_name : str
            Nome do jogador.
        save_path : str, optional
            Caminho para salvar.

        Returns
        -------
        Figure
            Objeto Figure com o mapa de chutes.
        """
        ...

    def plot_rapm_rankings(
        self,
        rapm_df: pd.DataFrame,
        top_n: int = 20,
        save_path: str | None = None,
    ) -> Any:
        """Grafico de barras horizontais com os rankings RAPM.

        Parameters
        ----------
        rapm_df : pd.DataFrame
            DataFrame com colunas 'player_name' e 'rapm_value'.
        top_n : int
            Numero de jogadores a exibir no topo e no fundo do ranking.
        save_path : str, optional
            Caminho para salvar a figura.

        Returns
        -------
        Figure
            Objeto Figure com o ranking RAPM.
        """
        ...

    def plot_synergy_graph(
        self,
        compatibility_df: pd.DataFrame,
        team: str | None = None,
        save_path: str | None = None,
    ) -> Any:
        """Grafo de sinergia/compatibilidade entre jogadores.

        Parameters
        ----------
        compatibility_df : pd.DataFrame
            DataFrame com colunas 'player_a', 'player_b' e 'score'
            representando a compatibilidade entre pares.
        team : str, optional
            Nome do time para o titulo.
        save_path : str, optional
            Caminho para salvar a figura.

        Returns
        -------
        Figure
            Objeto Figure com o grafo de sinergia.
        """
        ...
