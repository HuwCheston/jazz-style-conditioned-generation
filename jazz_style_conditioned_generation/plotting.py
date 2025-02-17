#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting classes, functions, and variables."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define constants
WIDTH = 18.8  # This is a full page width: half page plots will need to use 18.8 / 2
FONTSIZE = 18

ALPHA = 0.4
BLACK = '#000000'
WHITE = '#FFFFFF'

RED = '#FF0000'
GREEN = '#008000'
BLUE = '#0000FF'
YELLOW = '#FFFF00'
RGB = [RED, GREEN, BLUE]
CONCEPT_COLOR_MAPPING = {"Melody": "#ff0000ff", "Harmony": "#00ff00ff", "Rhythm": "#0000ffff", "Dynamics": "#ff9900ff"}

LINEWIDTH = 2
LINESTYLE = '-'
TICKWIDTH = 3
MARKERSCALE = 1.6
MARKERS = ['o', 's', 'D']
HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

DOTTED = 'dotted'
DASHED = 'dashed'

# Used for saving PNG images with the correct background
SAVE_KWS = dict(facecolor=WHITE, dpi=300)
SAVE_EXTS = ['svg', 'png', 'pdf']
# Keyword arguments to use when applying a grid to a plot
GRID_KWS = dict(color=BLACK, alpha=ALPHA, lw=LINEWIDTH / 2, ls=LINESTYLE)
# Used when adding a legend to an axis
LEGEND_KWS = dict(frameon=True, framealpha=1, edgecolor=BLACK)


class BasePlot:
    """Base plotting class from which all others inherit"""
    mpl.rcParams.update(mpl.rcParamsDefault)

    # These variables should all be overridden at will in child classes
    df = None
    fig, ax = None, None
    g = None

    def __init__(self, **kwargs):
        # Set fontsize
        plt.rcParams.update({'font.size': FONTSIZE})
        self.figure_title = kwargs.get('figure_title', 'baseplot')

    def _format_df(self, df: pd.DataFrame):
        return df

    def create_plot(self):
        """Calls plot creation, axis formatting, and figure formatting classes"""
        self._create_plot()
        self._format_ax()
        self._format_fig()

    def _create_plot(self) -> None:
        """This function should contain the code for plotting the graph"""
        return

    def _format_ax(self) -> None:
        """This function should contain the code for formatting the `self.ax` objects"""

        def _fmt(ax_):
            plt.setp(ax_.spines.values(), linewidth=LINEWIDTH, color=BLACK)
            ax_.tick_params(axis='both', width=TICKWIDTH, color=BLACK)

        if isinstance(self.ax, list):
            for ax in self.ax:
                _fmt(ax)
        elif isinstance(self.ax, np.ndarray):
            for ax in self.ax.flatten():
                _fmt(ax)
        else:
            _fmt(self.ax)

    def _format_fig(self) -> None:
        """This function should contain the code for formatting the `self.fig` objects"""
        self.fig.tight_layout()

    def close(self):
        """Alias for `plt.close()`"""
        plt.close(self.fig)

    def save_fig(self, fpath):
        """Saves a figure `fig` with filepath `fpath` using all extensions in `SAVE_EXTS`"""
        # Check that the directory we want to save the figure in exists
        root_dir = os.path.dirname(fpath)
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f'path {root_dir} does not exist')
        # For backwards compatibility: if we've added the extension in already, remove it
        if fpath.endswith(".png"):
            fpath = fpath.replace(".png", "")
        # Iterate through all the filetypes we want to use and save the plot
        for ext in SAVE_EXTS:
            self.fig.savefig(fpath + f'.{ext}', format=ext, **SAVE_KWS)
        # Close all figures at the end
        plt.close('all')


class BarPlotTiVoMetadataTagCounts(BasePlot):
    SCALE = True
    BAR_KWS = dict(edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, legend=False, zorder=10)

    def __init__(self, metadata: pd.DataFrame, tag_str: str = "genres", **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(metadata)
        self.tag_str = tag_str
        n_pianists = metadata['pianist'].nunique()
        self.fig, self.ax = plt.subplots(
            nrows=n_pianists // 2,
            ncols=2,
            sharex=True,
            sharey=self.SCALE,
            figsize=(WIDTH, n_pianists * 3)
        )

    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.SCALE:
            df['count'] = df.groupby('pianist')['count'].apply(lambda x: x / x.max()).values
        return df

    def _create_plot(self):
        for ax, (idx, grp) in zip(self.ax.flatten(), self.df.groupby('pianist')):
            sns.barplot(data=grp, x='value', y='count', ax=ax, **self.BAR_KWS)
            ax.set_title(idx)

    def _format_ax(self):
        for ax in self.ax.flatten():
            ax.tick_params(axis='x', rotation=90, labelbottom=True)
            ax.set(xlabel='', ylabel='')
            ax.grid(axis='y', zorder=0, **GRID_KWS)
        super()._format_ax()

    def _format_fig(self):
        self.fig.supxlabel(self.tag_str.title())
        self.fig.supylabel('Proportion of Tags')
        super()._format_fig()

    def save_fig(self, fpath):
        fpath += '_' + self.tag_str.lower()
        super().save_fig(fpath)
