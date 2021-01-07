import numpy
from data_structures import ConditionCollection, ResultsWithConditions
import plotly.graph_objects as go


class RegionProfile(object):

    def __init__(self, data, nrmlz_data, options): # raw_profile, layer_labels, colors=None):
        self._data = dict([(_col, ConditionCollection.from_pandas(data[_col]))
                           for _col in data.columns])
        self._norm_data = dict([(k, ConditionCollection.from_pandas(v))
                                for k, v in nrmlz_data.items()])
        self._norm_data[None] = ConditionCollection([ResultsWithConditions(1.0)])
        self.data_columns = data.columns
        self.possible_grp_cat = self._data[self.data_columns[0]].conditions()
        lbl_translator = options["Group translations"]
        self.possible_grp_lbl = [lbl_translator.get(_x, _x) for _x in self.possible_grp_cat]
        self.filter_values = dict([(lbl, self._data[self.data_columns[0]].labels_of(lbl))
                                   for lbl in self.possible_grp_cat])
        self._label_groups = options.get("Label groups", {})
        for k, v in self._label_groups.items():
            self.filter_values[k].extend(list(v.keys()))
        self._cols = options["Colors"]
        self._label_fmt = options["Labelling"]
        self._height = options["App"].get("Height", 700)
        self.fltr_vals_dicts = dict([(k, [dict([('label', _lbl), ('value', _lbl)]) for _lbl in lbls])
                                     for k, lbls in self.filter_values.items()])

        self.make_plot = {"Sankey": self.make_sankey,
                          "Sunburst": self.make_sunburst,
                          "Bar": self.make_horizontal_bar}

    def filter(self, data, fltr_spec, lenient=False):
        if len(fltr_spec) == 0:
            return data
        fltr_spec = dict(fltr_spec)
        for k, v in self._label_groups.items():
            if k in fltr_spec:
                fltr_spec[k] = numpy.hstack([v.get(_x, _x) for _x in fltr_spec[k]])
        return data.filter(filter_lenient=lenient, **fltr_spec)

    def normalize_by(self, filtered, nrmlz_cats):
        pool_conds = filtered.conditions()
        for category in nrmlz_cats:
            if category in pool_conds:
                pool_conds.remove(category)
        nrmlz_vals = filtered.pool(pool_conds, func=numpy.nansum)
        return filtered.extended_map(lambda x, y: x / y[0], [nrmlz_vals], ignore_conds=pool_conds)

    def _labeller_factory(self, grouping):
        grouping_sorted = [_x for _x in self._label_fmt["Hierarchy"]
                           if _x in grouping]
        lbl_format = self._label_fmt["Values"]

        def make_label(label_dict):
            if len(grouping_sorted):
                return " ".join(lbl_format.get(k, "{0}").format(label_dict[k])
                                for k in grouping_sorted)
            return "ALL"
        return make_label

    def _color_factory(self, grouping):
        grouping_sorted = [_x for _x in self._cols["Hierarchy"]
                           if _x in grouping]
        col_dict = self._cols["Values"]
        default_color = col_dict["_default"]
        all_color = col_dict.get("ALL", col_dict["_default"])

        def make_color(label_dict):
            if len(grouping_sorted):
                return col_dict.get(str(label_dict[grouping_sorted[0]]),
                                    default_color)
            return all_color
        return make_color

    def _overlap_matrix(self, grouping_from, grouping_to, filtered_data, normalization_data):
        label_dict_list_from = filtered_data.__recursive_accumulate__(grouping_from)
        label_dict_list_to = filtered_data.__recursive_accumulate__(grouping_to)
        out = numpy.zeros((len(label_dict_list_from), len(label_dict_list_to)), dtype=float)
        lbl_fun = self._labeller_factory(grouping_to)
        col_fun = self._color_factory(grouping_to)
        for i, label_dict_from in enumerate(label_dict_list_from):
            tmp_filtered = filtered_data.filter(**label_dict_from)
            tmp_normalize = normalization_data.filter(filter_lenient=True, **label_dict_from)
            for j, label_dict_to in enumerate(label_dict_list_to):
                out[i, j] = numpy.sum(tmp_filtered.get(**label_dict_to)) /\
                            (numpy.sum(tmp_normalize.get(filter_lenient=True, **label_dict_to)) + 1E-9)
        return out, list(map(lbl_fun, label_dict_list_to)),\
                    list(map(col_fun, label_dict_list_to))

    def make_groups(self, grouping_specs, filtered_data, normalization_data):
        label_dict_list = filtered_data.__recursive_accumulate__(grouping_specs[0])
        lbl_fun = self._labeller_factory(grouping_specs[0])
        col_fun = self._color_factory(grouping_specs[0])
        labels = [list(map(lbl_fun, label_dict_list))]
        str_mats = []
        colors = [list(map(col_fun, label_dict_list))]
        for grp_spec_fr, grp_spec_to in zip(grouping_specs[:-1], grouping_specs[1:]):
            new_mat, new_labels, new_cols = self._overlap_matrix(grp_spec_fr, grp_spec_to,
                                                                 filtered_data, normalization_data)
            labels.append(new_labels)
            str_mats.append(new_mat)
            colors.append(new_cols)
        return labels, str_mats, colors

    def dataframe_for_plotting(self, fltr_spec, grouping_spec, data_column, normalize_cats, normalize_column,
                               include_color=False):
        from pandas import DataFrame
        filtered_data = self.filter(self._data[data_column], fltr_spec)
        normalize_data = self.filter(self._norm_data[normalize_column], fltr_spec, lenient=True)
        if len(normalize_cats) > 0:
            filtered_data = self.normalize_by(filtered_data, normalize_cats)
        cat_groups = list(map(str, numpy.hstack(grouping_spec)))
        group_labels = [', '.join(grp) if len(grp) else "ALL"
                        for grp in grouping_spec]
        group_labels = ["{0}:".format(i) + _x for i, _x in enumerate(group_labels)]

        lbl_funcs = [self._labeller_factory(_grp) for _grp in grouping_spec]
        if len(lbl_funcs) == 0:
            lbl_funcs = [lambda x: "ALL"]
        if include_color:
            group_labels.append("_Color")
            if len(grouping_spec) >= 2:
                lbl_funcs.append(self._color_factory(grouping_spec[1]))
            else:
                lbl_funcs.append(lambda x: self._cols["Values"]["_default"])

        group_kwargs = filtered_data.__alternative_accumulate__(cat_groups)
        tmp_idxx = numpy.argsort([",".join(map(str, _x.values())) for _x in group_kwargs])
        group_kwargs = [group_kwargs[_i] for _i in tmp_idxx]
        nrmlz_kwargs = [dict([(k, v) for k, v in _kwargs.items() if k in normalize_data.conditions()])
                        for _kwargs in group_kwargs]
        out = []
        for grp_kw, nrml_kw in zip(group_kwargs, nrmlz_kwargs):
            out.append(
                [_lbl(grp_kw) for _lbl in lbl_funcs] + [numpy.sum(filtered_data.get(**grp_kw)) /
                                                        (numpy.sum(normalize_data.get(**nrml_kw)) + 1E-9)]
            )

        #out = [[_lbl(lcl_kwargs) for _lbl in lbl_funcs] + [numpy.sum(filtered_data.get(**lcl_kwargs))]
        #       for lcl_kwargs in group_kwargs]

        return DataFrame(out, columns=group_labels + ["_Value"])

    @staticmethod
    def get_plotly_labels(labels, colors):
        all_labels = []
        all_colors = []
        offsets = [0]
        for lbls, cols in zip(labels, colors):
            all_labels.extend(lbls)
            all_colors.extend(cols)
            offsets.append(len(all_labels))
        return all_labels, offsets, all_colors

    @staticmethod
    def get_plotly_links(str_mats, offsets, cols, threshold=0.0):
        link_dict = dict([("source", []), ("target", []), ("value", []), ("color", [])])
        for mat_index, mat in enumerate(str_mats):
            o1 = offsets[mat_index]
            o2 = offsets[mat_index + 1]
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if mat[i, j] > threshold:
                        link_dict['source'].append(o1 + i)
                        link_dict['target'].append(o2 + j)
                        link_dict['value'].append(mat[i, j])
                        link_dict['color'].append(cols[o1 + i].replace("1.0", "0.6")) # Adjusting transparency
        return link_dict

    def make_sankey(self, fltr_spec, grouping_spec, data_column, normalize_cats,
                    normalization_dataset=None, threshold=0.0):
        filtered_data = self.filter(self._data[data_column], fltr_spec)
        normalization_data = self.filter(self._norm_data[normalization_dataset], fltr_spec, lenient=True)
        if len(normalize_cats) > 0:
            filtered_data = self.normalize_by(filtered_data, normalize_cats)
        labels, str_mats, colors = self.make_groups(grouping_spec, filtered_data, normalization_data)
        all_labels, offset, lbl_colors = self.get_plotly_labels(labels, colors)
        label_dict = dict([("pad", 15), ("thickness", 20), ("line", dict([("color", "black"), ("width", 0.5)])),
                           ("label", all_labels), ("color", lbl_colors)])
        link_dict = self.get_plotly_links(str_mats, offset, lbl_colors, threshold=threshold)
        sankey = go.Sankey(node=label_dict,
                           link=link_dict)
        fwgt = go.FigureWidget(data=[sankey], layout=go.Layout())
        fwgt.layout.height = self._height
        return fwgt

    def make_sunburst(self, fltr_spec, grouping_spec, data_column, normalize_cats, normalization_dataset=None):
        import plotly.express as px
        dframe = self.dataframe_for_plotting(fltr_spec, grouping_spec, data_column, normalize_cats,
                                             normalize_column=normalization_dataset, include_color=False)
        group_labels = dframe.columns.values[:-1]

        sburst = px.sunburst(dframe, path=group_labels,
                             values="_Value")

        fwgt = go.FigureWidget(sburst, layout=go.Layout())
        fwgt.layout.height = self._height
        return fwgt

    def make_horizontal_bar(self, fltr_spec, grouping_spec, data_column, normalize_cats, normalization_dataset=None):
        dframe = self.dataframe_for_plotting(fltr_spec, grouping_spec, data_column, normalize_cats,
                                             normalize_column=normalization_dataset, include_color=True)
        group_labels = dframe.columns.values

        all_bars = []
        for label, label_df in dframe.groupby(group_labels[1]):
            all_bars.append(go.Bar(x=label_df["_Value"], y=label_df[group_labels[0]],
                                   name=label, marker={'color': label_df["_Color"]},
                                   orientation='h',
                                   text=[', '.join(map(str, _row)) for _row in label_df[group_labels[1:-1]].values],
                                   hoverinfo=['text', 'x']))

        fwgt = go.FigureWidget(data=all_bars, layout=go.Layout())
        fwgt.update_layout(barmode='stack')
        fwgt.layout.height = self._height
        return fwgt

    def make_empty(self):
        fwgt = go.FigureWidget(data=[])
        fwgt.layout.height = self._height
        return fwgt
