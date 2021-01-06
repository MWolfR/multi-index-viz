import numpy
import dash_core_components as dcc
import dash_html_components as html


str_fltr, str_f_val = 'filter-dropdown{0}', 'filter-value-dropdown{0}'


def make_filter_selectors(grouper_obj, default_filters, default_filter_vals, fltr_ctrl_types):
    fltr_opts_dict = [dict([('label', _lbl), ('value', _val)]) for _lbl, _val
                      in zip(grouper_obj.possible_grp_lbl,
                             grouper_obj.possible_grp_cat)]
    filter_selectors, filter_val_selectors = [], []
    for i in range(len(default_filters)):
        filter_selectors.append(dcc.Dropdown(
                                            options=fltr_opts_dict,
                                            value=default_filters[i],
                                            id=str_fltr.format(i)))
        if fltr_ctrl_types[i] == "Dropdown":
            filter_val_selectors.append(dcc.Dropdown(
                                                    options=[],
                                                    value=default_filter_vals[i],
                                                    multi=True,
                                                    id=str_f_val.format(i)))
        elif fltr_ctrl_types[i] == "RangeSlider":
            filter_val_selectors.append(dcc.RangeSlider(
                min=0,
                max=100,
                step=1,
                value=default_filter_vals[i],
                id=str_f_val.format(i)))
    return filter_selectors, filter_val_selectors


def make_grouping_selectors(group_obj, grouping):
    grp_dict = [dict([('label', _lbl), ('value', _val)]) for _lbl, _val in
                zip(group_obj.possible_grp_lbl, group_obj.possible_grp_cat)]

    grouping_selectors = [dcc.Dropdown(
                                        options=grp_dict,
                                        value=_grouping,
                                        multi=True,
                                        id="grouping-check{0}".format(i))
                          for i, _grouping in enumerate(grouping)]
    active_selectors = [dcc.Checklist(
                                      options=[dict([('label', 'active'), ('value', 1)])],
                                      value=[1],
                                      id="active-check{0}".format(i))
                        for i in range(len(grouping))]
    return grouping_selectors, active_selectors


def make_normalizer_selector(group_obj):
    grp_dict = [dict([('label', _lbl), ('value', _val)]) for _lbl, _val in
                zip(group_obj.possible_grp_lbl, group_obj.possible_grp_cat)]
    return dcc.Dropdown(
        options=grp_dict,
        value=[],
        multi=True,
        id="normalize-selector"
    )


def make_threshold_selector(min_val, max_val, use_step):
    marks_dict = dict([(v, "{0:3.2f}".format(v)) for v in numpy.linspace(min_val, max_val, 7)])
    thresh_selector = dcc.Slider(
        id='thresh-slider',
        min=min_val,
        max=max_val,
        value=min_val,
        step=use_step,
        marks=marks_dict
    )
    return thresh_selector


def make_plot_type_dropdown(default_val):
    opt_dict = [dict([('label', v), ('value', v)]) for v in ['Sankey', 'Bar', 'Sunburst']]
    return dcc.Dropdown(
        options=opt_dict,
        value=default_val,
        id="plot-type-dropdown")


def make_data_column_dropdown(column_names):
    default_val = column_names[0]
    opt_dict = [dict([('label', v), ('value', v)]) for v in column_names]
    return dcc.Dropdown(
        options=opt_dict,
        value=default_val,
        id="data-column-dropdown")


def html_layout(grouping_selectors, active_selectors, filter_selectors, filter_val_selectors,
                additional_elements, total_width=1000):
    n_groupings = len(grouping_selectors)
    n_filters = len(filter_selectors)
    adjectives = ['1st', '2nd', '3rd'] + ["{0}th".format(i) for i in range(4, n_groupings + 1)]
    adjectives = adjectives[:n_groupings]
    col_width = total_width / n_groupings

    fltr_rows = [html.Tr([html.Th("{0} grouping".format(adj), scope="col", style={"width": col_width})
                          for adj in adjectives]),
                 html.Tr([html.Td(html.Div([_x, _y])) for _x, _y in zip(active_selectors, grouping_selectors)])]
    col = 0
    i = 0
    n_cols = max(2, n_groupings)

    row_buffer = []
    for f_sel, f_v_sel in zip(filter_selectors, filter_val_selectors):
        row_buffer.extend([html.Td(html.Div([html.B("Filter category {0}:".format(i)), f_sel])),
                           html.Td(html.Div([html.B("Limit to:"), f_v_sel]))])
        i += 1
        col += 2
        if col + 2 > n_cols:
            fltr_rows.append(html.Tr(row_buffer))
            row_buffer = []
            col = 0
    if col + 2 > n_cols:
        fltr_rows.append(html.Tr(row_buffer))
        row_buffer = []
    for k, v in additional_elements.items():
        row_buffer.append(
            html.Td(html.Div([html.Label(k),
                              v]))
        )
    fltr_rows.append(html.Tr(row_buffer))

    grouping_div = html.Table(fltr_rows)
    return grouping_div
