import numpy
from data_structures import ConditionCollection
import plotly.graph_objects as go


class RegionProfile(object):

    def __init__(self, data, options): # raw_profile, layer_labels, colors=None):
        self._data = ConditionCollection.from_pandas(data)
        self._filtered = self._data
        self.possible_grp_cat = self._data.conditions()
        lbl_translator = options["Group translations"]
        self.possible_grp_lbl = [lbl_translator.get(_x, _x) for _x in self.possible_grp_cat]
        self.filter_values = dict([(lbl, self._data.labels_of(lbl)) for lbl in self.possible_grp_cat])
        self._label_groups = options.get("Label groups", {})
        for k, v in self._label_groups.items():
            self.filter_values[k].extend(list(v.keys()))
        self._cols = options["Colors"]
        self._label_fmt = options["Labelling"]
        self._height = options["App"].get("Height", 700)
        self.fltr_vals_dicts = dict([(k, [dict([('label', _lbl), ('value', _lbl)]) for _lbl in lbls])
                                     for k, lbls in self.filter_values.items()])
        self.make_plot = self.make_sankey
        if options["App"].get("Plot type", "Sankey") == "Sunburst":
            self.make_plot = self.make_sunburst

    def filter(self, fltr_spec):
        if len(fltr_spec) == 0:
            return self._data
        fltr_spec = dict(fltr_spec)
        for k, v in self._label_groups.items():
            if k in fltr_spec:
                fltr_spec[k] = numpy.hstack([v.get(_x, _x) for _x in fltr_spec[k]])
        return self._data.filter(**fltr_spec)

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
                return col_dict.get(label_dict[grouping_sorted[0]],
                                    default_color)
            return all_color
        return make_color

    def _overlap_matrix(self, grouping_from, grouping_to, filtered_data):
        label_dict_list_from = filtered_data.__recursive_accumulate__(grouping_from)
        label_dict_list_to = filtered_data.__recursive_accumulate__(grouping_to)
        out = numpy.zeros((len(label_dict_list_from), len(label_dict_list_to)), dtype=float)
        lbl_fun = self._labeller_factory(grouping_to)
        col_fun = self._color_factory(grouping_to)
        for i, label_dict_from in enumerate(label_dict_list_from):
            tmp_filtered = filtered_data.filter(**label_dict_from)
            for j, label_dict_to in enumerate(label_dict_list_to):
                out[i, j] = numpy.sum(tmp_filtered.get(**label_dict_to))
        return out, list(map(lbl_fun, label_dict_list_to)),\
                    list(map(col_fun, label_dict_list_to))

    def make_groups(self, grouping_specs, filtered_data):
        label_dict_list = filtered_data.__recursive_accumulate__(grouping_specs[0])
        lbl_fun = self._labeller_factory(grouping_specs[0])
        col_fun = self._color_factory(grouping_specs[0])
        labels = [list(map(lbl_fun, label_dict_list))]
        str_mats = []
        colors = [list(map(col_fun, label_dict_list))]
        for grp_spec_fr, grp_spec_to in zip(grouping_specs[:-1], grouping_specs[1:]):
            new_mat, new_labels, new_cols = self._overlap_matrix(grp_spec_fr, grp_spec_to, filtered_data)
            labels.append(new_labels)
            str_mats.append(new_mat)
            colors.append(new_cols)
        return labels, str_mats, colors

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

    def make_sankey(self, fltr_spec, grouping_spec, threshold=0.0):
        filtered_data = self.filter(fltr_spec)
        labels, str_mats, colors = self.make_groups(grouping_spec, filtered_data)
        all_labels, offset, lbl_colors = self.get_plotly_labels(labels, colors)
        label_dict = dict([("pad", 15), ("thickness", 20), ("line", dict([("color", "black"), ("width", 0.5)])),
                           ("label", all_labels), ("color", lbl_colors)])
        link_dict = self.get_plotly_links(str_mats, offset, lbl_colors, threshold=threshold)
        sankey = go.Sankey(node=label_dict,
                           link=link_dict)
        fwgt = go.FigureWidget(data=[sankey], layout=go.Layout())
        fwgt.layout.height = self._height
        return fwgt

    def make_sunburst(self, fltr_spec, grouping_spec, threshold=0.0):
        from pandas import DataFrame
        import plotly.express as px
        filtered_data = self.filter(fltr_spec)
        cat_groups = list(map(str, numpy.hstack(grouping_spec)))

        pool_conds = numpy.setdiff1d(filtered_data.conditions(), cat_groups).tolist()
        if len(pool_conds):
            tmp_dict = dict([(k, filtered_data.labels_of(k)) for k in filtered_data.conditions()])
            filtered_data = filtered_data.pool(pool_conds,
                                               func=numpy.sum)
            for k, v in tmp_dict.items():
                if len(v) == 1:
                    filtered_data.add_label(k, v[0])

        lbl_funcs = [self._labeller_factory(_grp) for _grp in grouping_spec]
        out = [[_lbl(entry.cond) for _lbl in lbl_funcs] + [entry.res]
               for entry in filtered_data.contents]
        dframe = DataFrame(out)

        sburst = px.sunburst(dframe, path=numpy.arange(len(lbl_funcs)).tolist(),
                             values=len(lbl_funcs))
        fwgt = go.FigureWidget(sburst, layout=go.Layout())
        fwgt.layout.height = self._height
        return fwgt
