import json
import dash
import pandas
import os
import numpy

import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output
from dash.dash import no_update

from grouper import RegionProfile
from sankey_plot import make_filter_selectors, make_grouping_selectors,\
    make_threshold_selector, make_plot_type_dropdown, html_layout, str_f_val, str_fltr


def read_config(config_fn):
    with open(config_fn, "r") as fid:
        options = json.load(fid)
    if isinstance(options["Data"], str):
        data_fn = os.path.join(os.path.split(os.path.abspath(config_fn))[0], options["Data"])
        with open(data_fn, "r") as fid:
            data = pandas.read_json(fid, orient="table")
    else:
        data = pandas.read_json(options["Data"], orient="table")
    if "Filter control types" not in options["App"]:
        options["App"]["Filter control types"] = ["Dropdown" for _ in options["App"]["Default filter"]]
    return data, options


def read_defaults(options):
    default_grouping = options["App"]["Default grouping"]
    default_filters, default_filter_vals = zip(*options["App"]["Default filter"])
    default_filters = [_x if len(_x) else None for _x in default_filters]
    min_val, max_val = options["App"]["Strength threshold"]
    use_step = (max_val - min_val) / 100
    return default_grouping, default_filters, default_filter_vals, min_val, max_val, use_step


def read_groupings(groups_and_actives):
    values_of_groups = groups_and_actives[::2]
    values_of_actives = groups_and_actives[1::2]
    return [grp for act, grp in zip(values_of_actives, values_of_groups)
            if len(act)]


def read_filters(fltr_vals_dicts, filters_and_values, fltr_ctrl_types):
    filters = filters_and_values[::2]
    f_values = filters_and_values[1::2]
    filter_spec = []
    for filter, values, f_type in zip(filters, f_values, fltr_ctrl_types):
        if filter is None:
            continue
        if f_type == "Dropdown":
            valids = fltr_vals_dicts.get(filter, [])
            values = [_val for _val in values if _val in valids]
        elif f_type == "RangeSlider":
            valids = fltr_vals_dicts.get(filter, numpy.array([]))
            values = [_val for _val in valids if _val >= values[0] and _val <= values[1]]
        if len(values):
            filter_spec.append((filter, values))
    return filter_spec


def main(config_fn, return_app=False):
    # Read configuration
    data, options = read_config(config_fn)
    default_grouping, default_filters, default_filter_vals, min_val, max_val, use_step = read_defaults(options)

    filter_spec = [(_fltr, _fval) for _fltr, _fval in
                   zip(default_filters, default_filter_vals)
                   if len(_fval) and _fltr is not None]
    fltr_ctrl_types = options["App"]["Filter control types"]
    # Set up initial state
    group_obj = RegionProfile(data, options)

    # Create app
    app = dash.Dash(__name__)#,
                    #external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])
    app.config["suppress_callback_exceptions"] = True

    # Create interactive components
    # fig = group_obj.make_plot[options["App"].get("Plot type", "Sankey")](filter_spec, default_grouping,
    #                                                                      threshold=min_val)
    fig = group_obj.make_empty()

    thresh_selector = make_threshold_selector(min_val, max_val, use_step)
    plot_type_dropdown = make_plot_type_dropdown(options["App"].get("Plot type", "Sankey"))
    inputs = [Input(thresh_selector.id, 'value'), Input(plot_type_dropdown.id, 'value')]

    filter_selectors, filter_val_selectors = make_filter_selectors(group_obj, default_filters,
                                                                   default_filter_vals,
                                                                   fltr_ctrl_types=fltr_ctrl_types)
    n_filters = len(filter_selectors)
    for f_sel, fv_sel in zip(filter_selectors, filter_val_selectors):
        inputs.extend([Input(f_sel.id, 'value'), Input(fv_sel.id, 'value')])

    grouping_selectors, active_selectors = make_grouping_selectors(group_obj, default_grouping)
    for grp_sel, act_sel in zip(grouping_selectors, active_selectors):
        inputs.extend([Input(grp_sel.id, 'value'), Input(act_sel.id, 'value')])

    # Create app layout
    controls_layout = html_layout(grouping_selectors, active_selectors, filter_selectors,
                                  filter_val_selectors, thresh_selector, plot_type_dropdown,
                                  total_width=1000)
    app.layout = html.Div([
        controls_layout,
        dcc.Graph(id='main-graph', figure=fig)
    ], style={'width': 1000})
    server = app.server

    # Main callback to update the figure
    @app.callback(
        Output('main-graph', 'figure'),
        inputs
    )
    def master_callback(new_thresh, plot_type, *args):
        filters = read_filters(group_obj.filter_values, args[:(2 * n_filters)], fltr_ctrl_types)
        groupings = read_groupings(args[(2 * n_filters):])
        return group_obj.make_plot[plot_type](filters, groupings, new_thresh)

    def filter_value_callback(i):
        @app.callback(
            Output(str_f_val.format(i), 'options'),
            [Input(str_fltr.format(i), 'value')]
        )
        def filter_callback_fun(new_filter_cat):
            new_options = group_obj.fltr_vals_dicts.get(new_filter_cat, [])
            return new_options

    def slider_range_callback(i):
        @app.callback(
            [Output(str_f_val.format(i), "min"),
            Output(str_f_val.format(i), "max"),
             Output(str_f_val.format(i), "step"),
             Output(str_f_val.format(i), "marks")],
            [Input(str_fltr.format(i), 'value')]
        )
        def slider_callback_fun(new_filter_cat):
            opts = group_obj.filter_values.get(new_filter_cat)
            if isinstance(opts[0], str):
                return no_update, no_update, no_update, no_update
            mn, mx = numpy.min(opts), numpy.max(opts)
            step = float(mx - mn) / 250
            if len(opts) <= 10:
                marks = dict([(int(mrk), str(mrk)) for mrk in opts])
            else:
                marks = dict([(int(v), str(v)) for v in numpy.linspace(mn, mx, 6)])
            print (marks)
            return mn, mx, step, marks

    [filter_value_callback(i) if fltr_ctrl_types[i] == "Dropdown"
     else slider_range_callback(i) for i in range(len(filter_selectors))]

    if return_app:
        return app
    return server


if __name__ == "__main__":
    import sys
    main(sys.argv[1], return_app=True).run_server(debug=False, use_reloader=True)
