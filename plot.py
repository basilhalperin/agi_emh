import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn
import matplotlib.gridspec as gridspec


BBOX_TO_ANCHOR = [1.25, 1.04]
BTA_BY_NUMCOL = {
    2: [1.44, -0.15],
    3: [2.4, -0.17],
}


def plot_df(
        df, title='',
        save=False, save_name='figs/tmp.png',
        plot_median=False,
        bbox_to_anchor=BBOX_TO_ANCHOR,
):
    fig, ax = plt.subplots(1)

    if plot_median:
        df.median(axis=1).plot(color='k', label='median', linewidth=4, ax=ax)

    df.plot(ax=ax)

    plt.legend(fontsize=12, bbox_to_anchor=bbox_to_anchor)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=14)

    if save:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.show()


def scatter_df(
        df, x_label, y_label, title='',
        date_range=True, fit_reg=True, scale_axes=1.05,
        save=False, save_name='figs/tmp.png',
        bbox_to_anchor=None,
):
    df = df[[x_label, y_label]]
    df = df.dropna()
    x = df[x_label]
    y = df[y_label]

    mx = df.unstack().max()
    plt.xlim(0, mx*scale_axes)
    plt.ylim(0, mx*scale_axes)

    label = 'OLS' if fit_reg else None
    seaborn.regplot(x=x, y=y, ci=False, fit_reg=fit_reg, line_kws={'alpha': 0.4, 'label': label})

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if date_range:
        start = max(x.dropna().index.min().year, y.dropna().index.min().year)
        end = min(x.dropna().index.max().year, y.dropna().index.max().year)
        title += '\n({0}-{1})'.format(start, end)
    plt.title(title, fontsize=14)

    plt.plot(range(0, int(mx*2)), color='k', alpha=0.4, label='45-degree', linestyle=':')
    plt.legend(fontsize=12, bbox_to_anchor=bbox_to_anchor)

    if save:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)

    plt.show()


"""
def scatter_r_vs_gdp(
        rs_by_horizon, rgdp_over_horizons, horizon, regplot=True,
        exclude=[], fortyfive=False, xlim=(-0.5, 4.75), ylim=(-0.5, 4.75),
        bbox_to_anchor=BBOX_TO_ANCHOR,
        save=False, save_base='r_figs/scatter_r_vs_gdp_'
):
    x = rs_by_horizon[horizon]
    y = rgdp_over_horizons[horizon]

    fig, ax = plt.subplots(1)

    stacked = pd.DataFrame()
    x = x[[country for country in x if country not in exclude]]
    for country in x:
        r_tmp = x[country].dropna()
        gdp_tmp = y[country].dropna()
        df = pd.DataFrame({'r': r_tmp, 'gdp_next_x': gdp_tmp})
        df = df.dropna()
        df['country'] = country
        plt.scatter(df['r'], df['gdp_next_x'], label=country)
        stacked = pd.concat([stacked, df], axis=0)

    if len(stacked) <= 1:
        plt.close()
        return stacked, None, fig, ax

    if fortyfive:
        plt.plot(plt.xlim(), plt.xlim(), linestyle='-', color='k', alpha=0.1,
                 scalex=False, scaley=False, label='45-degree')

    if regplot:
        reg = sm.OLS.from_formula(data=stacked, formula='gdp_next_x ~ r').fit()
        params = reg.params
        x = np.linspace(xlim[0], xlim[1])
        y = params['Intercept'] + params['r']*x

        plt.plot(x, y, linestyle='--', color='k', alpha=0.5,
                 scalex=False, scaley=False, label='OLS')
    else:
        reg = None

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.xlabel('{0}y real rate'.format(horizon), size=14)
    plt.ylabel('RGDP growth next {0} years'.format(horizon), size=14)
    plt.title('{0}y real rate vs. RGDP growth next {0} years'.format(horizon), size=14)
    plt.legend(ncol=1, loc='upper right', bbox_to_anchor=bbox_to_anchor)

    if save:
        plt.savefig(save_base+'{0}.png'.format(horizon), dpi=300, bbox_inches='tight')

    plt.show()

    return stacked, reg, fig, ax
"""


def scatter_r_vs_gdp(
        rs_by_horizon, rgdp_over_horizons,
        horizon_map={10: (0, 0)}, dim=(1, 1), figsize=(5, 5),
        regplot=True,
        exclude=[], fortyfive=False, xlim=(-0.5, 4.75), ylim=(-0.5, 4.75),
        bbox_to_anchor=[1, -0.45],
        supxlabel='Real rate over horizon',
        supylabel='Real GDP growth over succeeding horizon',
        suptitle='Real rate vs. future real GDP growth',
        titleweight=None,
        subplot_titles=False,
        legend_ncol=1,
        save=False, save_name='r_figs/scatter_delete.png'
):
    num_rows = dim[0]
    num_cols = dim[1]

    gs = gridspec.GridSpec(num_rows, num_cols)
    fig = plt.figure(figsize=figsize)

    axes = {}
    horizons = horizon_map.keys()
    for h in horizons:
        g = gs[horizon_map[h][0], horizon_map[h][1]]
        axes[h] = plt.subplot(g)

    regs = {}
    stackeds = {}

    for h in axes:
        x = rs_by_horizon[h]
        y = rgdp_over_horizons[h]

        ax = axes[h]

        stacked = pd.DataFrame()
        x = x[[c for c in x if c not in exclude]]
        for country in x:
            r_tmp = x[country].dropna()
            gdp_tmp = y[country].dropna()
            df = pd.DataFrame({'r': r_tmp, 'gdp_next_x': gdp_tmp})
            df = df.dropna()
            df['country'] = country
            ax.scatter(df['r'], df['gdp_next_x'], label=country)
            stacked = pd.concat([stacked, df], axis=0)

        stackeds[h] = stacked

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if fortyfive:
            ax.plot(plt.xlim(), plt.xlim(), linestyle='-', color='k', alpha=0.1,
                    scalex=False, scaley=False, label='45-degree')

        if regplot and len(stacked) > 1:
            reg = sm.OLS.from_formula(data=stacked, formula='gdp_next_x ~ r').fit()
            params = reg.params
            x = np.linspace(xlim[0], xlim[1])
            y = params['Intercept'] + params['r']*x
            regs[h] = reg

            ax.plot(x, y, linestyle='--', color='k', alpha=0.5,
                    scalex=False, scaley=False, label='OLS')
        else:
            reg = None
        regs[h] = reg

        if subplot_titles:
            ax.set_title('{0}-year horizon'.format(h), size=12)

        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)

    fig.supxlabel(supxlabel, size=14)
    fig.supylabel(supylabel, size=14)
    fig.suptitle(suptitle, size=16, weight=titleweight)
    fig.tight_layout()
    #plt.legend(bbox_to_anchor=bbox_to_anchor, ncol=legend_ncol, fontsize=12)

    # legend is annoying because want to take intersection of labels
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    labels_to_lines = {}
    i = 0
    for i in range(len(lines_labels)):
        (lines, labels) = lines_labels[i]
        for j in range(len(labels)):
            label = labels[j]
            if label not in labels_to_lines:
                labels_to_lines[label] = lines[j]
        i += 1

    lines = list(labels_to_lines.values())
    labels = list(labels_to_lines.keys())
    fig.legend(lines, labels, bbox_to_anchor=bbox_to_anchor, ncol=legend_ncol, fontsize=12)

    if save:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')

    return stackeds, regs


def plot_r_vs_gdp(
        r_by_horizon, rgdp_ahead, horizon, start='1970',
        bta=BTA_BY_NUMCOL[2], figsize=None,
        num_col=2,
        save=False, save_name='r_figs/ts_r_vs_gdp_{0}.png'
):
    num_countries = len(r_by_horizon.columns)
    num_rows = math.ceil(num_countries/num_col)
    num_columns = min(num_col, num_countries)    # 1 col if 1 country

    if figsize is None:
        figsize = (9, max(3*num_rows, 4))

    gs = gridspec.GridSpec(num_rows, num_columns)
    fig = plt.figure(figsize=figsize)
    i = 0
    j = 0
    for country in r_by_horizon:
        ax = plt.subplot(gs[i, j])
        df = pd.DataFrame({
            'r': r_by_horizon[country],
            'GDP {0}y ahead'.format(horizon): rgdp_ahead[country]
        })
        df.index.name = None
        df = df.loc[start:]
        df.plot(ax=ax, legend=None)

        ax.set_title(country, size=14)
        ax.set_xticks(ax.get_xticks(), size=12)
        ax.set_yticks(ax.get_yticks(), size=12)
        j += 1
        if j%num_columns == 0:
            j = 0
            i += 1
    fig.suptitle('{0}y real rate vs. GDP growth {0}y ahead'.format(horizon), size=16)
    fig.tight_layout()
    legend_ax = plt.subplot(gs[num_rows-1, 0])
    legend_ax.legend(fontsize=12, ncol=2, bbox_to_anchor=bta)

    if save:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
