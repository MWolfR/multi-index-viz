from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import zip
from builtins import map
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import numpy
# import os
import logging
# import h5py
# import progressbar
# import inspect
from builtins import isinstance

logger = logging.getLogger('simulationStruc')


def identity(x):
    return x


class ResultsWithConditions(object):

    def __init__(self, res, *args, **kwargs):
        self.cond = {}
        if len(args) == 0:
            self._res = res
            self._in_memory = True
        else:
            self._in_memory = False
            self._func = args[0]
            self._arg = res
            if len(args) > 1:
                self._load = bool(args[1])
            else:
                self._load = True
        for k, v in list(kwargs.items()):
            if k.startswith('_'):
                continue
            if isinstance(v, numpy.ndarray) or isinstance(v, list):
                self.cond[k] = tuple(v)
            else:
                self.cond[k] = v

    @property
    def res(self):
        if not self._in_memory:
            if not self._load:
                return self._func(self._arg)
            self._res = self._func(self._arg)
            self._in_memory = True
        return self._res

    def update(self, other_conds):
        for k, v in list(other_conds.items()):
            if k.startswith('_'):
                continue
            if isinstance(v, numpy.ndarray) or isinstance(v, list):
                self.cond[k] = tuple(v)
            else:
                self.cond[k] = v

    def check(self, **kwargs):
        for k, v in list(kwargs.items()):
            if self.cond[k] != v:
                return False
        return True

    def into_h5(self, h5obj, write_func=None, **kwargs):
        if write_func is None:
            import simProjectAnalysis.io
            write_func = simProjectAnalysis.io.result_into_h5
        for k, v in list(self.cond.items()):
            try:
                h5obj.attrs.create(k, v)
            except TypeError:
                logger.warn("Label for condition %s has to be converted to str" % k)
                h5obj.attrs.create(k, str(v))
        write_func(self.res, h5obj, **kwargs)

    @staticmethod
    def from_h5(h5obj, read=True, read_func=None):
        if read_func is None:
            import simProjectAnalysis.io
            read_func = simProjectAnalysis.io.result_from_h5
        cond = dict(list(h5obj.attrs.items()))
        #cond.pop('_class')
        if read:
            res = read_func(h5obj)
            return ResultsWithConditions(res, **cond)
        else:
            return ResultsWithConditions(h5obj, read_func,
                                         **cond)


class ConditionCollection(object):

    def __init__(self, items):
        self.contents = items
        self.__pbar_min_num__ = 10000
        self.__metadata__()

    def __pbar__(self):
        return iter

    def __metadata__(self):
        self.cond_map = {}
        for idx, c in self.__pbar__()(enumerate(self.contents)):
            for k in list(c.cond.keys()):
                cond_val = c.cond[k]
#                if isinstance(cond_val, numpy.ndarray) or isinstance(cond_val, list):
#                    cond_val = tuple(cond_val)
                self.cond_map.setdefault(k, {}).setdefault(cond_val, []).append(idx)

    def __recursive_accumulate__(self, splt_at, **kwargs):
        if len(splt_at) == 0:
            return [kwargs.copy()]
        splt = splt_at[0]
        lbls = self.labels_of(splt)
        cp_kw = kwargs.copy()
        ret = []
        for lbl in lbls:
            cp_kw[splt] = lbl
            ret.extend(self.__recursive_accumulate__(splt_at[1:], **cp_kw))
        return ret

    def __alternative_accumulate__(self, splt_at, **kwargs):
        if len(splt_at) == 0:
            return [{}]
        idxx = numpy.zeros(len(self.contents), dtype=int)
        fac = 1
        used_keys = []
        used_facs = []
        for cond in splt_at:
            cm = self.cond_map[cond]
            kk = list(cm.keys())
            used_keys.append(kk)
            for i, _k in enumerate(kk):
                idxx[cm[_k]] += fac * i
            fac *= len(kk)
            used_facs.append(fac)
        div_facs = numpy.hstack([1, used_facs])
        uidx = numpy.unique(idxx)
        final_conds = []
        for cond, div_fac, mod_fac, kk in zip(splt_at, div_facs, used_facs, used_keys):
            final_conds.append([kk[int(_i)] for _i in old_div(numpy.mod(uidx, mod_fac), div_fac)])
        return [dict(list(zip(splt_at, zipped_conds))) for zipped_conds in zip(*final_conds)]

    def __pool_combinations__(self, conds, force_no_empty_conds=False):
        if numpy.prod([len(self.labels_of(_cond)) for _cond in conds]) == 1:
            logger.warn('Nothing to pool along %s.' % str(conds))
            #return None
        cond_lst = list(self.cond_map.keys())
        if isinstance(conds, list):
            [cond_lst.remove(x) for x in conds]
        else:
            cond_lst.remove(conds)
            conds = [conds]
        cond_lst = [_c for _c in cond_lst if len(self.labels_of(_c)) > 1]
        if len(cond_lst) == 0:
            return [{}]
        srt = numpy.argsort([numpy.mean(list(map(len, list(self.cond_map[k].values()))))
                             for k in cond_lst])[-1::-1]
        cond_lst = [cond_lst[i] for i in srt]
        if (numpy.prod([len(self.labels_of(_x)) for _x in cond_lst]) > len(self.contents)) or force_no_empty_conds:
            logger.info("Getting remaining conditions... Strawberry version")
            all_kwargs = self.__alternative_accumulate__(cond_lst)
        else:
            logger.info("Getting remaining conditions... Classic version")
            all_kwargs = self.__recursive_accumulate__(cond_lst)
        logger.info("Found %d combinations of conditions" % len(all_kwargs))

        return all_kwargs

    def idx(self, **kwargs):
        idx = list(range(len(self.contents)))
        for k, v in list(kwargs.items()):
            if k not in self.cond_map:
                return numpy.array([], dtype=int)
            if v.__hash__ is not None:
                idx = numpy.intersect1d(idx, self.cond_map[k].get(v, []))
            else:
                idx = numpy.intersect1d(idx,
                                        numpy.hstack([self.cond_map[k].get(_v, [])
                                                     for _v in v]))
        return idx

    def merge(self, other):
        '''Merge these results with other results, keeping individual condition labels'''
        offset = len(self.contents)
        self.contents.extend(other.contents)
        for k, mp in list(other.cond_map.items()):
            for k2, v in list(mp.items()):
                self.cond_map.setdefault(k, {}).setdefault(k2, []).extend([vv + offset for vv in v])

    def append(self, item):
        '''Append another result with conditions'''
        idx = len(self.contents)
        self.contents.append(item)
        for k, v in list(item.cond.items()):
            self.cond_map.setdefault(k, {}).setdefault(v, []).append(idx)

    def extend(self, items):
        idx = len(self.contents)
        self.contents.extend(items)
        for o, item in enumerate(items):
            for k, v in list(item.cond.items()):
                self.cond_map.setdefault(k, {}).setdefault(v, []).append(idx + o)

    def all_labels_of(self, key):
        '''Return for _all_ contained results the specified condition label'''
        return [x.cond[key] for x in self.contents]

    def labels_of(self, key):
        '''Return the existing (unique) labels of the specified condition'''
        return sorted(self.cond_map[key].keys())

    def conditions(self):
        '''Return all existing conditions'''
        return list(self.cond_map.keys())

    def add_label(self, new_label, new_val):
        for x in self.__pbar__()(self.contents):
            if new_label in list(x.cond.keys()):
                raise Exception("Condition " + new_label + " already assigned!")
            x.cond[new_label] = new_val
        self.cond_map[new_label] = {new_val: list(range(len(self.contents)))}

    def remove_label(self, label):
        if isinstance(label, list):
            [self.remove_label(_label) for _label in label]
            return self
        if label not in self.cond_map:
            return self
        for x in self.__pbar__()(self.contents):
            x.cond.pop(label, None)
        self.cond_map.pop(label, None)
        return self

    def __legacy_get__(self, **kwargs):
        return [x.res for x in self.contents if x.check(**kwargs)]

    def get(self, **kwargs):
        return [self.contents[i].res for i in self.idx(**kwargs)]

    def get_x_y(self, conds, **kwargs):
        idxx = self.idx(**kwargs)
        if not isinstance(conds, list):
            conds = [conds]
        if len(idxx) == 0:
            return [() for _ in range(len(conds) + 1)]
        return list(zip(*[[self.contents[i].cond[_c]
                      for _c in conds] + [self.contents[i].res]
                     for i in idxx]))

    def get_x_y_sorted(self, conds, **kwargs):
        data = self.get_x_y(conds, **kwargs)
        if len(data) == 1:
            return data
        idxx = [_m * numpy.digitize(_x, bins=numpy.unique(_x)) for _x, _m in
                zip(data[:-1], numpy.cumprod(list(map(len, data[:-1]))))]
        idxx = numpy.argsort(numpy.vstack(idxx).sum(axis=0))
        return [[_d[_idx] for _idx in idxx] for _d in data]

    def get2(self, **kwargs):
        retval = self.get(**kwargs)
        if len(retval) == 1:
            return retval[0]
        return retval

    def __legacy_map__(self, func, **kwargs):
        ret_val = ConditionCollection([])
        for x in self.contents:
            if x.check(**kwargs):
                ret_val.append(ResultsWithConditions(func(x.res), **x.cond))
        return ret_val

    def map(self, func, **kwargs):
        ret_val = ConditionCollection([])
        for i in self.idx(**kwargs):
            ret_val.append(ResultsWithConditions(func(self.contents[i].res),
                                                 **self.contents[i].cond))
        return ret_val

    def split(self, split_at):
        if not isinstance(split_at, list):
            split_at = [split_at]
        if numpy.prod([len(self.labels_of(_x)) for _x in split_at]) > len(self.contents):
            logger.info("Getting remaining conditions... Strawberry version")
            all_kwargs = self.__alternative_accumulate__(split_at)
        else:
            logger.info("Getting remaining conditions... Classic version")
            all_kwargs = self.__recursive_accumulate__(split_at)
        CV = [ConditionCollection([self.contents[i] for i in
                                   self.idx(**loc_kwargs)])
              for loc_kwargs in all_kwargs]
        CC = [[loc_kwargs[_k] for _k in split_at]
              for loc_kwargs in all_kwargs]
        return (CC, CV)

    def __legacy_split__(self, split_at):
        if isinstance(split_at, list):
            if len(split_at) > 1:
                cond_vals = sorted(self.cond_map[split_at[0]].keys())
                splt_c = []
                splt_v = []
                for v in cond_vals:
                    tc, tv = ConditionCollection([self.contents[i] for i in
                                                  self.idx(**{split_at[0]: v})]).split(split_at[1:])
                    splt_v.extend(tv)
                    splt_c.extend([[v] + _tc for _tc in tc])
                return (splt_c, splt_v)
            else:
                split_at = split_at[0]
        cond_vals = sorted(self.cond_map[split_at].keys())
        return ([[x] for x in cond_vals],
                [ConditionCollection([self.contents[i] for i in
                                      self.idx(**{split_at: v})])
                 for v in cond_vals])

    def __legacy_pool__(self, conds, func=None):
        if func is None:
            func = lambda x: x
        cond_lst = list(self.cond_map.keys())
        if isinstance(conds, list):
            [cond_lst.remove(x) for x in conds]
        else:
            cond_lst.remove(conds)
        st_cond = [[]]
        st_idxx = [set(range(len(self.contents)))]
        for cond in cond_lst:
            st_cond = [_conds + [k] for _conds in st_cond
                       for k in list(self.cond_map[cond].keys())]
            st_idxx = [_idxx.intersection(self.cond_map[cond][k])
                       for _idxx in st_idxx for k in list(self.cond_map[cond].keys())]
        newres = [ResultsWithConditions(func([self.contents[i].res for i in _idxx]),
                                        **dict(list(zip(cond_lst, _cond)))) for _cond, _idxx
                  in zip(st_cond, st_idxx) if len(_idxx) > 0]
        return ConditionCollection(newres)

    def pool(self, conds, func=None, xy=False, force_no_empty_conds=False, filter_empty=True):
        all_kwargs = self.__pool_combinations__(conds, force_no_empty_conds=force_no_empty_conds)
        if all_kwargs is None:
            if func is None:
                return self
            else:
                return self.map(lambda x: x)
        if func is None:
            func = lambda x: x
        ret = []
        for lcl_kwargs in all_kwargs:
            if xy:
                got = self.get_x_y(conds, **lcl_kwargs)
                if (len(got[0]) > 0) or not filter_empty:
                    ret.append(ResultsWithConditions(func(*got), **lcl_kwargs))
            else:
                got = self.get(**lcl_kwargs)
                if (len(got) > 0) or not filter_empty:
                    ret.append(ResultsWithConditions(func(got), **lcl_kwargs))
        return ConditionCollection(ret)

    def transform(self, conds, func=None, xy=False, force_no_empty_conds=False, filter_empty=True):
        all_kwargs = self.__pool_combinations__(conds, force_no_empty_conds=force_no_empty_conds)
        if all_kwargs is None:
            if func is None:
                return self
            else:
                return self.map(lambda x: x)
        if func is None:
            func = lambda x: x
        ret = []
        for lcl_kwargs in all_kwargs:
            if xy:
                got = self.get_x_y(conds, **lcl_kwargs)
                if (len(got[0]) > 0) or not filter_empty:
                    for res, spec_conds in func(*got):
                        lcl_copy = lcl_kwargs.copy()
                        lcl_copy.update(spec_conds)
                        ret.append(ResultsWithConditions(res, **lcl_copy))
            else:
                got = self.get(**lcl_kwargs)
                if (len(got) > 0) or not filter_empty:
                    for res, spec_conds in func(got):
                        lcl_copy = lcl_kwargs.copy()
                        lcl_copy.update(spec_conds)
                        ret.append(ResultsWithConditions(res, **lcl_copy))
        return ConditionCollection(ret)

    def unpool(self, func):
        '''TODO: All other functions are non-destructive, while this one is destructive.
        INCONSISTENT'''
        for _ in self.__pbar__()(range(len(self.contents))):
            v = self.contents.pop(0)
            for res, newconds in func(v):
                base_cond = v.cond.copy()
                base_cond.update(newconds)
                self.contents.extend([ResultsWithConditions(res, **base_cond)])
        self.__metadata__()

    def __legacy_extended_map__(self, func, others, func_kwargs={}, ignore_conds=[], iterate_inner=False, **kwargs):
        ret_val = ConditionCollection([])
        for x in self.contents:
            if x.check(**kwargs):
                matcher = x.cond.copy()
                [matcher.pop(ignore, None) for ignore in ignore_conds]
                matching = [o.filter(**matcher) for o in others]
                matching_vals = [m.get() for m in matching]
                if iterate_inner:
                    new_res = [ResultsWithConditions(func(x.res, *mtch, **func_kwargs), **matcher)
                               for mtch in zip(*matching_vals)]
                    for i, mtch in enumerate(matching):
                        for ignore in ignore_conds:
                            inner_conds = mtch.all_labels_of(ignore)
                            for j, inner in enumerate(inner_conds):
                                new_res[j].cond['arg' + str(i+1) + '_' + ignore] = inner
                    for ignore in ignore_conds:
                        for j in range(len(new_res)):
                            new_res[j].cond['arg0_' + ignore] = x.cond[ignore]
                    ret_val.contents.extend(new_res)
                else:
                    #matching = [m[0] if len(m) == 1 else m for m in matching]
                    ret_val.append(ResultsWithConditions(func(x.res, *matching_vals, **func_kwargs), **matcher))
        return ret_val

    def extended_map(self, func, others, func_kwargs={}, ignore_conds=[], iterate_inner=False, **kwargs):
        ret_val = ConditionCollection([])
        for i in self.idx(**kwargs):
            x = self.contents[i]
            matcher = x.cond.copy()
            [matcher.pop(ignore, None) for ignore in ignore_conds]
            matching = [o.filter(**matcher) for o in others]
            matching_vals = [m.get() for m in matching]
            if iterate_inner:
                new_res = [ResultsWithConditions(func(x.res, *mtch, **func_kwargs), **matcher)
                           for mtch in zip(*matching_vals)]
                for i, mtch in enumerate(matching):
                    for ignore in ignore_conds:
                        inner_conds = mtch.all_labels_of(ignore)
                        for j, inner in enumerate(inner_conds):
                            new_res[j].cond['arg' + str(i+1) + '_' + ignore] = inner
                for ignore in ignore_conds:
                    for j in range(len(new_res)):
                        new_res[j].cond['arg0_' + ignore] = x.cond[ignore]
                ret_val.contents.extend(new_res)
            else:
                #matching = [m[0] if len(m) == 1 else m for m in matching]
                ret_val.append(ResultsWithConditions(func(x.res, *matching_vals, **func_kwargs), **matcher))
        ret_val.__metadata__()
        return ret_val

    def filter(self, **kwargs): #TODO: Make filter such that it does not have to read results that are not in memory
        return self.map(identity, **kwargs)

    def to_pandas(self, pandas_type='auto', **kwargs):
        """
        Turn the data represented into a pandas.DataFrame or pandas.Series
        :param pandas_type: (str): specify 'series' to return a pandas.Series or 'dataframe' for DataFrame.
        Default: 'auto', which uses a Series for one-dimensional data (one scalar per combination of conditions) and
        a DataFrame else
        :param kwargs: Further kwargs to be handed to the pandas constructor. For example: columns=['col1', 'col2',...]
        in the case of pandas_type='dataframe'.
        :return: A pandas.Series or pandas.DataFrame object with a MultiIndex representing the conditions and the
        data field representing the results.

        Note that this function does not check whether the data is fit to be represented in this way. If the results
        represented by this structure are non-numerical it is likely to result in an exception.
        """

        from pandas import MultiIndex, DataFrame, Series
        index_labels = self.conditions()
        index = [[_x.cond[_c] for _c in index_labels] for _x in self.contents]
        multi_index = MultiIndex.from_tuples(index, names=index_labels)
        data = [_x.res for _x in self.contents]

        # Function to help decide whether dataframe or series
        def scalar_or_one(x):
            if not hasattr(x, '__len__'):
                return 2
            return int(len(x) == 1)

        if pandas_type == 'auto':
            if numpy.all(map(scalar_or_one, data)):
                pandas_type = 'series'
            else:
                pandas_type = 'dataframe'
        if pandas_type.lower() == 'series':
            tp = Series
            if numpy.all(numpy.array(list(map(scalar_or_one, data))) == 1):
                data = numpy.hstack(data)
        elif pandas_type.lower() == 'dataframe':
            tp = DataFrame
        else:
            raise ValueError("Unknown pandas type: {tp}".format(tp=pandas_type))

        return tp(data, index=multi_index, **kwargs)

    @staticmethod
    def from_pandas(pandas_obj):
        """
        Create a ConditionCollection from a pandas.Series or pandas.DataFrame object.
        the input object MUST be indexed by a pandas.MultiIndex which will be used to decide the conditions.
        Further, the MultiIndex MUST have assigned names to its columns.
        The values of the object will be turned into the results.
        :param pandas_obj: Input
        :return: A ConditionCollection object
        """
        multi_index = pandas_obj.index
        from pandas import MultiIndex, DataFrame, Series
        assert isinstance(multi_index, MultiIndex)
        index_labels = multi_index.names
        assert not numpy.any([_x is None for _x in index_labels])
        results = []
        if isinstance(pandas_obj, Series):
            for conds, res in pandas_obj.iteritems():
                cond_dict = dict(list(zip(index_labels, conds)))
                results.append(ResultsWithConditions(res, **cond_dict))
        else:
            for conds, res in pandas_obj.iterrows():
                cond_dict = dict(list(zip(index_labels, conds)))
                results.append(ResultsWithConditions(res.values, **cond_dict))
        return ConditionCollection(results)

