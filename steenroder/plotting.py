import numpy as np
import plotly.graph_objs as gobj
from plotly.colors import DEFAULT_PLOTLY_COLORS as COLS


def _compute_bounds(barcode):
    barcode_no_dims = np.concatenate(barcode)
    posinfinite_mask = np.isposinf(barcode_no_dims)
    neginfinite_mask = np.isneginf(barcode_no_dims)
    max_val = np.max(np.where(posinfinite_mask, -np.inf, barcode_no_dims))
    min_val = np.min(np.where(neginfinite_mask, np.inf, barcode_no_dims))
    parameter_range = max_val - min_val
    extra_space_factor = 0.02
    has_posinfinite_death = np.any(posinfinite_mask[:, 1])
    if has_posinfinite_death:
        posinfinity_val = max_val + 0.1 * parameter_range
        extra_space_factor += 0.1
    else:
        posinfinity_val = None
    has_neginfinite_birth = np.any(neginfinite_mask[:, 0])
    if has_neginfinite_birth:
        neginfinity_val = min_val - 0.1 * parameter_range
        extra_space_factor += 0.1
    else:
        neginfinity_val = None
    extra_space = extra_space_factor * parameter_range
    min_val_display = min_val - extra_space
    max_val_display = max_val + extra_space

    return min_val_display, max_val_display, posinfinity_val, neginfinity_val


def plot_diagrams(barcode, steenrod_barcode, k=None, kind=None, tex=False,
                  plotly_params=None):
    """Plot a regular persistence barcode and a Steenrod barcode as diagrams on
    a common birth-death plane.

    Parameters
    ----------
    barcode : list of ndarray
        The persistence barcode to plot. For each dimension ``d``, a 2D array
        with 2 columns, containing birth-death pairs in degree ``d``, to be used
        as coordinates in the two-dimensional plot.

    steenrod_barcode : list of ndarray
        The (relative) Sq^k-barcode to plot. For each dimension ``d``, a 2D
        array with 2 columns, containing the birth-death pairs of Steenrod bars
        in degree ``d``, to be used as coordinates in the two-dimensional plot.

    k : int or None, optional, default: ``None``
        Positive integer defining the cohomology operation Sq^k that was
        performed to obtain `steenrod_barcode`. Only used for labelling.

    kind : ``"R"`` | ``"A"`` or None, optional, default: ``None``
        Whether the barcodes to be plotted come from absolute or relative
        cohomology barcodes.

    tex :  bool, optional, default: ``False``
        (Experimental!) Whether to display a version of the legend rendered with
        LaTeX and MathJax.

    plotly_params : dict or None, optional, default: ``None``
        Custom parameters to configure the plotly figure. Allowed keys are
        ``"traces"`` and ``"layout"``, and the corresponding values should be
        dictionaries containing keyword arguments as would be fed to the
        :meth:`update_traces` and :meth:`update_layout` methods of
        :class:`plotly.graph_objects.Figure`.

    Returns
    -------
    fig : :class:`plotly.graph_objects.Figure` object
        Figure representing the persistence diagram and Steenrod diagram.

    """
    def _connect_st_label(st_label, h_label, tex):
        if tex:
            st_h_label = r"$" + st_label + r" \cap " + h_label + r"$"
            h_label = r"$" + h_label + r"$"
        else:
            st_h_label = st_label + " in " + h_label
            h_label = h_label

        return h_label, st_h_label

    if k is not None:
        st_label = r"\mathrm{img}" + rf"(Sq^{{{k}}})" if tex else f"im(Sq^{k})"
    else:
        st_label = r"\mathrm{img}(Sq^{k})" if tex else "im(Sq^k)"
    kind = kind.lower()
    if kind == "a":
        legend_title = "Absolute Cohomology"
        h_subscript = r"_{A}" if tex else ""
    elif kind == "r":
        legend_title = "Relative Cohomology"
        h_subscript = r"_{R}" if tex else ""
    else:
        legend_title = "Cohomology"
        h_subscript = r"" if tex else ""
    homology_dimensions = range(max(len(barcode), len(steenrod_barcode)))

    min_val_display, max_val_display, posinfinity_val, neginfinity_val = \
        _compute_bounds(barcode + steenrod_barcode)

    fig = gobj.Figure()
    fig.add_trace(gobj.Scatter(
        x=[min_val_display, max_val_display],
        y=[min_val_display, max_val_display],
        mode="lines",
        line={"dash": "dash", "width": 1, "color": "black"},
        showlegend=False,
        hoverinfo="none"
        ))

    for i, dim in enumerate(homology_dimensions):
        h_label = (rf"\mathcal{{H}}^{{{dim}}}" if tex else f"H^{dim}") + \
                  h_subscript
        h_label, st_h_label = _connect_st_label(st_label, h_label, tex)
        for label, symbol, ms, bc in ([st_h_label, "diamond", 10, steenrod_barcode],
                                      [h_label, "circle", 8, barcode]):
            subbc = bc[dim].copy()
            unique, inverse, counts = np.unique(
                subbc, axis=0, return_inverse=True, return_counts=True
                )
            hovertext = [
                f"{tuple(unique[unique_row_index][:2])}" +
                (
                    f", multiplicity: {counts[unique_row_index]}"
                    if counts[unique_row_index] > 1 else ""
                )
                for unique_row_index in inverse
                ]
            births = subbc[:, 0]
            if neginfinity_val is not None:
                births[np.isneginf(births)] = neginfinity_val
            deaths = subbc[:, 1]
            if posinfinity_val is not None:
                deaths[np.isposinf(deaths)] = posinfinity_val
            fig.add_trace(gobj.Scatter(
                x=births, y=deaths, name=label,
                mode="markers", marker_color=COLS[i], marker_symbol=symbol,
                marker_line={"width": 1}, marker_size=ms,
                hoverinfo="text", hovertext=hovertext,
                showlegend=True
            ))

    fig.update_layout(
        width=500,
        height=500,
        xaxis1={
            "title": "Birth",
            "side": "bottom",
            "type": "linear",
            "range": [min_val_display, max_val_display],
            "autorange": False,
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
            },
        yaxis1={
            "title": "Death",
            "side": "left",
            "type": "linear",
            "range": [min_val_display, max_val_display],
            "autorange": False, "scaleanchor": "x", "scaleratio": 1,
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
            },
        plot_bgcolor="white",
        legend={
            "y": 0.01,
            "x": 0.6,
        },
        legend_title=legend_title if not tex else None,
        font_family="Serif",
        font_size=14
    )

    # Add a horizontal dashed line for points with infinite death
    if posinfinity_val is not None:
        fig.add_trace(gobj.Scatter(
            x=[min_val_display, max_val_display],
            y=[posinfinity_val, posinfinity_val],
            mode="lines",
            line={"dash": "dash", "width": 0.5, "color": "black"},
            showlegend=True,
            name=r"+" + u"\u221E" + 25 * " ",
            hoverinfo="none"
        ))

    # Add a vertical dashed line for points with negative infinite birth
    if neginfinity_val is not None:
        fig.add_trace(gobj.Scatter(
            x=[neginfinity_val, neginfinity_val],
            y=[min_val_display, max_val_display],
            mode="lines",
            line={"dash": "dash", "width": 0.5, "color": "black"},
            showlegend=True,
            name="-" + u"\u221E" + 25 * " ",
            hoverinfo="none"
        ))

    # Update traces and layout according to user input
    if plotly_params:
        fig.update_traces(plotly_params.get("traces", None))
        fig.update_layout(plotly_params.get("layout", None))

    return fig
