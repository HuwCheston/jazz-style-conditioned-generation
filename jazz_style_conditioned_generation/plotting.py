#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting classes, functions, and variables."""

import os
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from jazz_style_conditioned_generation import utils

# Define constants
WIDTH = 18.8  # This is a full page width: half page plots will need to use 18.8 / 2
FONTSIZE = 18

ALPHA = 0.4
BLACK = '#000000'
GREY = "#808080"
WHITE = '#FFFFFF'

RED = '#FF0000'
GREEN = '#008000'
BLUE = '#0000FF'
CYAN = '#00FFFF'
MAGENTA = '#FF00FF'
YELLOW = '#FFFF00'
RGB = [RED, GREEN, BLUE]
CMY = [CYAN, MAGENTA, YELLOW]

LINEWIDTH = 2
LINESTYLE = '-'
TICKWIDTH = 3
MARKERSCALE = 1.6
MARKERS = ['o', 's', 'D']
HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
BARWIDTH = 0.7

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


class PointPlotVocabSizeCustomLoss(BasePlot):
    PPLOT_KWS = dict(
        estimator="mean", color=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE,
        capsize=LINEWIDTH / 10, err_kws=dict(linewidth=LINEWIDTH), x="vocab_size",
    )

    def __init__(self):
        super().__init__()
        self.df = pd.read_csv(os.path.join(utils.get_project_root(), "references/vocab_size_vs_loss.csv"))
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            nrows=1, ncols=2, sharex=True, sharey=False, figsize=(WIDTH, WIDTH // 3)
        )

    def _create_plot(self) -> None:
        sns.pointplot(data=self.df, y="decoded_length_loss", errorbar="sd", ax=self.ax1, **self.PPLOT_KWS)
        sns.pointplot(data=self.df, y="mean_loss", errorbar=None, ax=self.ax2, **self.PPLOT_KWS)

    def _format_ax(self) -> None:
        self.ax1.set(
            xlabel="Vocabulary Size",
            ylabel=r"$\dfrac{\text{sum}(\text{loss})}{\text{len}(\text{sequence}_{\text{raw}})}$"
        )
        self.ax2.set(
            xlabel="Vocabulary Size",
            ylabel=r"$\dfrac{\text{sum}(\text{loss})}{\text{len}(\text{sequence})}$"
        )
        for ax in [self.ax1, self.ax2]:
            ax.set_xticks(ax.get_xticks(), ["Raw"] + [i.get_text() for i in ax.get_xticklabels()][1:], rotation=90)
            ax.grid(axis='y', zorder=0, **GRID_KWS)
            plt.setp(ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
            ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)

    def _format_fig(self) -> None:
        self.fig.tight_layout()


class HeatmapPerformerGenreCounts(BasePlot):
    """Heatmap showing total counts of tracks for all genres and pianists"""

    HEATMAP_KWS = dict(
        linecolor=GREY, fmt="", square=True, linewidths=LINEWIDTH / 3, linestyles=LINESTYLE, cbar=None, cmap="Blues"
    )

    def __init__(self, df, **kwargs):
        super().__init__(figure_title="heatmap_performer_genre_counts", **kwargs)
        self.df = self._format_df(df)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH))

    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        from jazz_style_conditioned_generation.data.conditions import INCLUDE

        # Create the dataframe
        fmt = (
            df.groupby("performer")
            .value_counts()
            .reset_index(drop=False)
            .pivot(columns="genre", index="performer")
            .fillna(0)
            .astype(int)
            .droplevel(0, axis=1)
        )
        # Fill in missing genres with columns of zeros
        for genre in INCLUDE["genres"]:
            if genre not in fmt.columns:
                fmt[genre] = 0
        return (
            fmt.reindex(sorted(fmt.columns, reverse=True), axis=1)
            .astype(int)
            .transpose()
        )

    def _create_plot(self) -> None:
        mask = self.df.copy()
        mask[mask == 0] = ""
        sns.heatmap(data=self.df, mask=self.df == 0, annot=mask, ax=self.ax, **self.HEATMAP_KWS)

    def add_dual_axis(self):
        # Get the sums for both x and y-axis
        total_x, total_y = self.df.sum(axis=0).values, self.df.sum(axis=1).values
        # Add dual x axis
        x_top = self.ax.secondary_xaxis("top")
        x_top.set_xticks(np.arange(0.5, len(total_x)), labels=total_x, rotation=90)
        # Add dual y axis
        y_top = self.ax.secondary_yaxis("right")
        y_top.set_yticks(np.arange(0.5, len(total_y)), labels=total_y)
        # Set aesthetics
        for ax_ in [x_top, y_top]:
            plt.setp(ax_.spines.values(), linewidth=LINEWIDTH, color=BLACK)
            ax_.tick_params(width=0.)

    def _format_ax(self):
        self.add_dual_axis()
        # Set spine visibility and width to create a border
        for spine in self.ax.spines.values():
            spine.set_visible(True)
        # Set aesthetics for main colorbar axis
        plt.setp(self.ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
        self.ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
        self.ax.set(ylabel="Genre", xlabel="Pianist")


class BarPlotPianistGenreCount(BasePlot):
    """Creates a barplot showing numbers of albums for all genres + pianists"""

    BAR_KWS = dict(edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, legend=False, zorder=10, width=BARWIDTH)

    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(WIDTH, WIDTH // 2), sharex=False, sharey=False)

    def _create_plot(self) -> None:
        for res, ax, color in zip(self.df, self.ax.flatten(), [RED, BLUE]):
            res_fmt = pd.Series(res).sort_values()
            sns.barplot(data=res_fmt, ax=ax, color=color, **self.BAR_KWS)

    def _format_ax(self):
        for ax, name in zip(self.ax.flatten(), ["Pianist", "Genre"]):
            ax.set(xlabel=name)
            ax.tick_params(axis='x', rotation=90, labelbottom=True)
            ax.grid(axis='y', zorder=0, **GRID_KWS)
        self.ax[0].set(ylabel="Number of albums")
        super()._format_ax()


class BarPlotGenrePCENPS(BasePlot):
    """Creates a barplot showing pitch-class entropy and notes-per-second for all genres"""

    BAR_KWS = dict(edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, legend=False, zorder=10, width=BARWIDTH)

    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(WIDTH, WIDTH / 2), sharex=False, sharey=False)
        self.df = df

    def _create_plot(self):
        for var, ax_, color in zip(["pce", "nps"], self.ax.flatten(), [RED, BLUE]):
            sub = self.df.sort_values(by=var).reset_index(drop=True)
            sns.barplot(data=sub, x="name", y=var, ax=ax_, color=color, **self.BAR_KWS)

    def _format_ax(self):
        for ax_, name in zip(self.ax.flatten(), ["Sliding pitch class entropy", "Notes per second"]):
            ax_.set_xticklabels(ax_.get_xticklabels(), rotation=90)
            ax_.set(xlabel="Genre", ylabel=name)
            ax_.grid(axis='y', zorder=0, **GRID_KWS)
        self.ax[0].set_ylim(2., 2.35)
        super()._format_ax()


class BarPlotWeightDistribution(BasePlot):
    """Plot the number of album tags with different weights assigned by TiVo"""

    BAR_KWS = dict(
        edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, legend=False, zorder=10, color=GREEN, width=BARWIDTH
    )

    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(df)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH // 2, WIDTH // 3))

    def _format_df(self, df) -> pd.DataFrame:
        return (
            pd.DataFrame([df])
            .transpose()
            .sort_index()
            .rename(columns={0: "norm"})
            .reset_index(drop=False)
        )

    def _create_plot(self):
        sns.barplot(data=self.df, x="index", y="norm", ax=self.ax, **self.BAR_KWS)

    def _format_ax(self):
        self.ax.set(xlabel="Assigned weight", ylabel="Number of tags")
        super()._format_ax()


class BarPlotGroupedGenreCounts(BasePlot):
    BAR_KWS = dict(
        edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, legend=False, zorder=10, color=GREEN, width=BARWIDTH
    )

    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(df)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH // 3))

    def _format_df(self, df):
        return (
            pd.DataFrame([df])
            .transpose()
            .rename(columns={0: "count"})
            .sort_values(by="count")
            .reset_index(drop=False)
            .rename(columns={"index": "genre"})
        )

    def _create_plot(self):
        sns.barplot(data=self.df, x="count", y="genre", ax=self.ax, **self.BAR_KWS)

    def _format_ax(self):
        self.ax.set(xlabel="Recordings", ylabel="Genre category")
        self.ax.grid(axis='x', zorder=0, **GRID_KWS)
        # self.ax.axvline(x=4462, ymin=0, ymax=1, linestyle=DASHED, linewidth=LINEWIDTH, color=BLACK)
        # self.ax.text(4400, 10, "Total number of recordings", rotation=90, ha="center", va="center")
        super()._format_ax()


class BarPlotMeanClampScore(BasePlot):
    BAR_KWS = dict(edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, legend=True, zorder=10)
    ERROR_KWS = dict(fmt='none', color=BLACK, capsize=LINEWIDTH * 2, zorder=100000, linewidth=LINEWIDTH)
    N_BOOT = 1000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(kwargs.get("res"))
        self.fig, self.ax = plt.subplots(
            nrows=2, ncols=1, sharex=False, sharey=True, figsize=(WIDTH, WIDTH)
        )

    def _format_df(self, df: pd.DataFrame):
        from jazz_style_conditioned_generation.data.conditions import INCLUDE

        cond_toks = set(i["token"] for i in df)
        res = []
        for cond_tok in cond_toks:
            for gen_real in ["generated", "real"]:
                gen = [i for i in df if i["token"] == cond_tok and i["type"] == gen_real][0]
                sims = gen["cosine_sims"]
                gen_boots = [np.mean(np.random.choice(sims, size=len(sims), replace=True)) for _ in range(self.N_BOOT)]
                res_tok = dict(
                    type=gen_real.title(),
                    token=cond_tok,
                    mean=np.mean(sims),
                    std=np.std(sims),
                    low=abs(np.mean(sims) - np.percentile(gen_boots, 2.5)),
                    high=abs(np.mean(sims) - np.percentile(gen_boots, 97.5)),
                    is_pianist=cond_tok in INCLUDE["pianist"]
                )
                res.append(res_tok)
        return pd.DataFrame(res)

    def _format_ax(self):
        for ax_ in self.ax.flatten():
            ax_.set_xticks(ax_.get_xticks(), labels=ax_.get_xticklabels(), rotation=90)
            ax_.grid(axis='y', zorder=0, **GRID_KWS)
            sns.move_legend(ax_, loc="upper right", title="", **LEGEND_KWS)
            ax_.set(xlim=[-0.5, max(ax_.get_xlim()) - 0.75], ylabel="Average CLaMP-3 Score")
        super()._format_ax()

    def _create_plot(self):
        for (idx, grp), ax_ in zip(self.df.groupby("is_pianist", as_index=False), self.ax.flatten()):
            grp = grp.sort_values(by="mean", ascending=False).reset_index(drop=True)
            sns.barplot(data=grp, x="token", y="mean", hue="type", ax=ax_, **self.BAR_KWS)
            ax_.set(xlabel="Pianist" if idx else "Genre")
            # Get the x positions of the bars
            bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in ax_.patches]  # Bar midpoints
            bar_heights = [patch.get_height() for patch in ax_.patches]  # Bar heights
            # Loop through the bars and add the error bars using iterrows
            for i, (_, row) in enumerate(grp.iterrows()):
                # Add error bars
                ax_.errorbar(bar_positions[i], bar_heights[i], yerr=row["std"], **self.ERROR_KWS)


class BarPlotSubjectiveQuality(BasePlot):
    """Bar plot for quality (enjoyment, diversity, AI-generated) questions asked during listening test"""

    N_BOOT = 10000
    BAR_KWS = dict(
        edgecolor=BLACK,
        linewidth=LINEWIDTH,
        linestyle=LINESTYLE,
        legend=True,
        zorder=10,
        errorbar=('ci', 95),
        err_kws=dict(
            zorder=10000,
            color=BLACK,
            linewidth=LINEWIDTH
        ),
        capsize=0.2,
        n_boot=N_BOOT,
        x="variable",
        y="value",
    )

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)

        self.df_cond, self.df_genre = self._format_df(df)
        self.fig, self.ax = plt.subplots(1, 2, figsize=(WIDTH, WIDTH / 3))

    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format the dataframe for `quality` questions (preference, diversity, is_ml)"""
        kind_map = {"real": "Ground Truth", "clamp": "DPO-P", "noclamp": "No DPO-P"}
        gen_map = {
            "avantgardejazz": "Avant-Garde",
            "straightaheadjazz": "Straight-Ahead",
            "traditionalearlyjazz": "Traditional"
        }
        df["kind"] = df["condition_type"].map(kind_map)
        df["genre"] = df["actual_genre"].map(gen_map)
        by_condition = (
            df.melt(id_vars="kind", value_vars=["fit", "preference", "diversity", "is_ml"])
            .dropna()
            .reset_index(drop=True)
        )
        by_genre = (
            df.melt(id_vars="genre", value_vars=["fit", "preference", "diversity", "is_ml"])
            .dropna()
            .reset_index(drop=True)
        )
        return by_condition, by_genre

    def _create_plot(self):
        _ = sns.barplot(data=self.df_cond, hue="kind", ax=self.ax[0], palette=RGB, **self.BAR_KWS)
        _ = sns.barplot(data=self.df_genre, hue="genre", ax=self.ax[1], palette=CMY, **self.BAR_KWS)

    def _format_ax(self):
        for ax, tit in zip(self.ax.flatten(), ["Condition", "Genre"]):
            ax.set(
                title=tit,
                xlabel="Question",
                xticklabels=[r"Fit $\uparrow$", r"Liking $\uparrow$", r"Creativity $\uparrow$",
                             r"AI-Generated $\downarrow$"],
                ylabel="Mean response",
                ylim=(0, 5.7)  # gives us a bit of headroom
            )
            ax.grid(axis="y", zorder=0, **GRID_KWS)
            sns.move_legend(ax, title="", loc="upper right", **LEGEND_KWS)
        super()._format_ax()


class BarPlotSubjectiveSimilarity(BasePlot):
    """Bar plot for similarity question asked during listening test"""
    BAR_KWS = dict(
        edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, legend=False, zorder=10, width=BARWIDTH / 2
    )
    TEXT_KWS = dict(va='center', ha='center', zorder=10000, fontsize=FONTSIZE)
    CONDITIONS = [
        'CLaMP/target+nCLaMP/target',
        'CLaMP/target+real/target', 'CLaMP/target+real/wrong', 'CLaMP/target+CLaMP/wrong',
        'nCLaMP/target+nCLaMP/wrong', 'nCLaMP/target+real/target', 'nCLaMP/target+real/wrong'
    ]

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.df_non_norm = None
        if kwargs.get("use_toy_data", False):
            self.df = self._get_toy_data()
            self.df_non_norm = self.df
        else:
            self.df = self._format_df(df)
        self.fig, self.ax = plt.subplots(ncols=2, nrows=1, figsize=(WIDTH, WIDTH // 2), sharey=False, sharex=True)

    def _get_toy_data(self) -> pd.DataFrame:
        """Gets toy data in the correct format, for use before we have real data"""

        def getter():
            rand = [random.random() for _ in range(3)]
            a_wins, ties, b_wins = [(i / sum(rand)) * 100 for i in rand]
            return dict(a_wins=a_wins, ties=ties, b_wins=b_wins)

        is_clamp = [False, True, True, True, False, False, False]
        toy_data = [{"kind": k, **getter(), "is_clamp": ic} for k, ic in zip(self.CONDITIONS, is_clamp)]
        return pd.DataFrame(toy_data).set_index("kind")

    @staticmethod
    def get_winner(row: pd.Series) -> str:
        """Determine the winner for each row"""
        if row["similar"] == "a":
            return row["a"]
        elif row["similar"] == "b":
            return row["b"]
        else:
            return "tie"

    @staticmethod
    def summarize_group(grp: pd.DataFrame) -> pd.Series:
        """Group and count results"""
        a, b = grp.name.split("+")
        counts = grp["winner"].value_counts()
        return pd.Series({
            "kind": grp.name,
            "a_wins": counts.get(a, 0),
            "b_wins": counts.get(b, 0),
            "ties": counts.get("tie", 0),
        })

    @staticmethod
    def df_as_proportion(to_prop: pd.DataFrame) -> pd.DataFrame:
        """Convert row values to percentages"""
        return to_prop.div(to_prop.sum(axis=1), axis=0) * 100

    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        mapper = {
            "gen_match_clamp": "CLaMP/target",
            "gen_nomatch_clamp": "CLaMP/wrong",
            "gen_match_noclamp": "nCLaMP/target",
            "gen_nomatch_noclamp": "nCLaMP/wrong",
            "real_match": "real/target",
            "real_nomatch": "real/wrong",
            "tie": "tie"
        }
        # Apply winner function and map to standardized labels
        df["winner"] = df.apply(self.get_winner, axis=1).map(mapper)
        result_df = (
            df.groupby("type")
            .apply(self.summarize_group)
            .reset_index(drop=True)
            .set_index("kind")
        )
        self.df_non_norm = result_df.copy()
        prop = self.df_as_proportion(result_df).reset_index(drop=False)
        prop["is_clamp"] = prop["kind"].apply(lambda x: True if "nCLaMP" not in x else False)
        return prop.set_index("kind")[["a_wins", "ties", "b_wins", "is_clamp"]]

    def _create_plot(self):
        self.df = self.df[self.df.index != "CLaMP/target+nCLaMP/target"]
        for (idx, grp), ax_ in zip(self.df.groupby("is_clamp"), self.ax.flatten()):
            ax_ = grp.plot(kind='barh', stacked=True, ax=ax_, **self.BAR_KWS)
            # Add custom text labels
            for i, row in grp.iterrows():
                prev_above = False
                a, b = str(i).split("+")
                cum_width = 0
                for s, col in zip([a, "No Preference", b], ["a_wins", "ties", "b_wins"]):
                    if "nCLaMP/" in s:
                        s = s.replace("nCLaMP/", "Model, ")
                    elif "CLaMP/" in s:
                        s = s.replace("CLaMP/", "Model, ")
                    if "real/" in s:
                        s = s.replace("real/", "Real, ")
                    if "target" in s:
                        s = s.replace("target", "match")
                    value = row[col]
                    y = ax_.get_yticks()[list(grp.index).index(i)]
                    if value < 25:
                        if prev_above:
                            y -= 0.3
                            prev_above = False
                        else:
                            prev_above = True
                            y += 0.3
                        color = BLACK
                        s = f"{s}\n({value:.1f}%)"
                    else:
                        color = WHITE
                        if value < 35:
                            s = f"{s}\n({value:.1f}%)"
                        else:
                            s = f"{s} ({value:.1f}%)"
                        prev_above = False

                    ax_.text(x=cum_width + value / 2, y=y, s=s, color=color, **self.TEXT_KWS)
                    cum_width += value
            ax_.set_title("Fine-tune/DPO-P" if idx else "Fine-tune")

    def _format_ax(self):
        for ax_ in self.ax.flatten():
            ax_.set(
                yticklabels=["" for _ in range(len(ax_.get_yticklabels()))],
                ylabel="",
                xlabel="Voting Rate (%)",
                xlim=(0, 100)
            )
            ax_.grid(axis='x', zorder=0, **GRID_KWS)
        super()._format_ax()


class HeatmapSubjectiveAccuracy(BasePlot):
    HMAP_KWS = dict(
        annot=True, cbar=False, square=True, linewidths=LINEWIDTH,
        linestyle=LINESTYLE, linecolor=BLACK, cmap="rocket",
        vmin=0, vmax=15
    )
    HMAP_TYPES = ["DPO-P", "No DPO-P", "Ground truth"]

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.mapper = {
            "avantgardejazz": "Avant-Garde",
            "straightaheadjazz": "Straight-Ahead",
            "traditionalearlyjazz": "Traditional"
        }
        self.labels = sorted(df["actual_genre"].map(self.mapper).unique())
        self.df, self.accs = self._format_df(df)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=len(self.df), sharex=True, sharey=False,
                                         figsize=(WIDTH, WIDTH / 2.5))

    def _format_df(self, df: pd.DataFrame):
        all_cts = []
        all_accs = []
        for idx, ct in df.groupby("condition_type", sort=True):
            y_true = ct["actual_genre"].values
            y_pred = ct["predicted_genre"].values
            all_cts.append(confusion_matrix(y_true, y_pred, labels=sorted(self.mapper.keys())))
            all_accs.append((y_true == y_pred).mean() * 100)
        return all_cts, all_accs

    def _create_plot(self):
        for a, ct in zip(self.ax.flatten(), self.df):
            sns.heatmap(data=ct, ax=a, **self.HMAP_KWS)

    def _format_ax(self):
        for g, ct_type, acc in zip(self.ax.flatten(), self.HMAP_TYPES, self.accs):
            title = f"{ct_type}\nAccuracy: {round(acc, 1)}%"
            g.set(title=title, xticklabels=self.labels, yticklabels=self.labels)
            g.set_xticklabels(self.labels, rotation=90)
            g.set_yticklabels(self.labels, rotation=0)
        super()._format_ax()

    def _format_fig(self) -> None:
        self.fig.supylabel("True label")
        self.fig.supxlabel("Predicted label")
        super()._format_fig()


if __name__ == "__main__":
    pp = PointPlotVocabSizeCustomLoss()
    pp.create_plot()
    pp.save_fig(os.path.join(utils.get_project_root(), "outputs/figures/vocab_size/pointplot_vocab_size_custom_loss"))
