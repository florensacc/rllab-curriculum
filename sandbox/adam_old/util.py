import multiprocessing as mp
import copy


class TimingRecord(object):

    def __init__(self, keys_flat, to_share_dicts, stamp_keys, report_headers, n_proc=2):
        """
        For timing: only needs to be assembled at end, so make a local dict for easy writing
        while iterating, and a shared dict for writing at the end, both same hierarchy. For
        final reporting, want each entry to be a list with a value for each process.
        (In the end, hopefully this approach is easier than using a manager dict, which has 
        limited capabilities vs standard dicts.)

        'keys_flat' and 'stamp_keys' will define what times are recorded.  Entries in 'stamp_keys'
        need to match the timestamps that are fed to the 'collect_times()' method (i.e. within 
        the algo train() method) for a given mode. The modes correspond to the top-level keys of
        'stamp_keys' and 'keys_flat', and currently only 'ser' and 'par' are supported.
        """
        self.keys_flat = keys_flat
        self.stamp_keys = stamp_keys
        self.to_share_dicts = to_share_dicts
        self.report_headers = report_headers
        self.n_proc = n_proc



        # Build the hierarchy here for clarity: make lists of keys, 
        # where nested dicts will be denoted by 'times_' appended 
        # to the front of the name of the list.
        #
        # EXAMPLE: (like in define_timing() method of TRPO_par)
        #
        # (RE-PASTE SOMETHING HERE.)

        self.times_flat = {}
        self.times_flat_s = {}
        for k,v in keys_flat.iteritems():
            self.times_flat[k] = self._init_flat_dict(v)
            if to_share_dicts[k]:
                self.times_flat_s[k] = self._init_flat_dict(v, mode='multi-shared')
 

    def _init_flat_dict(self, keys_flat_single, mode='single'):
        init_flat_dict = self._build_empty_flat_dict(keys_flat_single)
        self._fill_flat_dict(init_flat_dict, mode=mode)
        self._link_hierarchy(init_flat_dict)
        return init_flat_dict


    def _build_empty_flat_dict(self, keys_flat_single):
        empty_flat_dict = {}
        for k,v in keys_flat_single.iteritems():
            sub_dict = {}
            for v1 in v:
                sub_dict[v1] = None
            empty_flat_dict[k] = sub_dict
        return empty_flat_dict


    def _fill_flat_dict(self, dict_to_fill, mode='single'):
        for k,v in dict_to_fill.iteritems():
            for k1 in v:
                if mode=='multi-shared':
                    dict_to_fill[k][k1] = [mp.RawValue('d') for _ in xrange(self.n_proc)]
                elif mode=='multi':
                    dict_to_fill[k][k1] = [0.] * self.n_proc
                elif mode=='single':
                    dict_to_fill[k][k1] = 0.
                else:
                    dict_to_fill[k][k1] = 0.


    def _link_hierarchy(self, flat_dict):
        """
        Only flat_dict layout is used for writing the times,
        but the hierarchical dict layout is used for reporting.
        """
        for k in flat_dict:
            for k1,v1 in flat_dict.iteritems():
                if 'times_'+k in v1:
                    flat_dict[k1]['times_'+k] = flat_dict[k]
                    break


    def record_timestamps(self, timestamps, dict_name, stamp_name):
        """
        For use by any method/function which has access to this timing object.
        """
        times_master = self.times_flat[dict_name]
        stamp_keys = self.stamp_keys[stamp_name]
        for k,v in stamp_keys.iteritems():
            for k1,v1 in v.iteritems():
                times_master[k][k1] += timestamps[v1[0]] - timestamps[v1[1]] 


    def record_time_dicts(self, time_dicts, dict_names, key_names):
        """
        Inputs as lists.
        """
        assert(len(time_dict) == len(dict_name) and len(dict_name) == len(key_name))
        for t_dict, d_name, k_name in zip(time_dicts, dict_names, key_names):
            self.record_time_dict(t_dict, d_name, k_name)


    def record_time_dict(self, time_dict, dict_name, key_name):
        """
        For use in recording timing data returned from function/methods which do not
        have access to this timing object (i.e. can't use record_timestamps()).
        """
        for k,v in time_dict.iteritems():
            self.times_flat[dict_name][key_name][k] += v


    def write_to_shared(self, rank, dict_name):
        shared_dict = self.times_flat_s[dict_name]
        private_dict = self.times_flat[dict_name]
        for k,v in private_dict.iteritems():
            for k1,v1 in v.iteritems():
                if not isinstance(v1, dict): # (need this 'if' because already linked)
                    shared_dict[k][k1][rank].value = v1            


    def _private_copy_dict(self, dict_name):
        """
        So it can be treated the same as regular values, i.e. don't have to write
        var.value to retrieve value.
        """
        shared_dict = self.times_flat_s[dict_name]
        # copy.deepcopy() didn't work with shared c types, so do this:
        priv_dict = self._init_flat_dict(self.keys_flat[dict_name], mode='multi')
        for k,v in shared_dict.iteritems():
            for k1,v1 in v.iteritems():
                if not isinstance(v1, dict):
                    if isinstance(v1, list):
                        for i,val in enumerate(v1):
                            priv_dict[k][k1][i] = val.value
                    else:
                        priv_dict[k][k1] = v1.value
        return priv_dict



    ########################
    ###  REPORT WRITING  ###
    ########################


    def _write_dict_report(self, data_dict, at_top=True, fmt_str='6.2f'):
        """
        Calls itself recursively to write all levels of a hierarchical data_dict.
        (Writing order branched for deep levels.)
        """
        report = ''
        if not at_top:
            if not any(isinstance(v, dict) for _,v in data_dict.iteritems()):
                # (Don't write the lowest level by itself: already written as sub-time.)
                return report

        for k in data_dict:
            if (not isinstance(data_dict[k], dict)) and (at_top or 'times_'+k in data_dict):
                report += '\t{0:10}\t'.format(k+':')
                report += self._write_report_values(data_dict[k], fmt_str='<'+fmt_str)
                if 'times_' + k in data_dict:
                    for k1 in data_dict['times_'+k]:
                        if not isinstance(data_dict['times_'+k][k1], dict):
                            report += '\t  {0:10}\t'.format(k1+':') # (small indent)
                            report += self._write_report_values(data_dict['times_'+k][k1], fmt_str)

        report += '\n'

        for k in data_dict:
            if isinstance(data_dict[k], dict):
                report += self._write_dict_report(data_dict[k], at_top=False, fmt_str=fmt_str)

        return report


    def _write_report_values(self, values, fmt_str):
        value_report = ''
        if not isinstance(values, list):
            values = [values]
        for v in values:
            value_report += '{0:{1}}\t'.format(v, fmt_str)
        value_report += '\n'
        return value_report
                

    def generate_report(self, dict_names=[]):
        """
        dict_names is an (ordered) list of which timing dicts to print.
        If a dict_name appears in the shared times, the values are
        taken from the shared version of the dict.
        """
        n_proc = self.n_proc
        proc_label = '\t    proc:\t'
        for p in range(n_proc): proc_label += '{:^6d}\t'.format(p)
        proc_label += '\n\t\t\t' + ' ----\t' * n_proc + '\n'

        report = ''

        for name in dict_names:
            report += '\n\t--- Timing Report: ' + self.report_headers[name] + ' ---\n'
            if name in self.times_flat_s:
                report += proc_label
                priv_dict = self._private_copy_dict(name)
                report += self._write_dict_report(priv_dict['_main'])
            else:
                report += self._write_dict_report(self.times_flat[name]['_main'])

        return report


    def print_report(self, dict_names=[]):
        report = self.generate_report(dict_names)
        print report


    ###############################################################
    ###  LEFTOVER: (incomplete) from purely hierarchical dicts  ###
    ###############################################################

    # def _init_dict(self, keys_flat, mode='single'):
    #     empty_dict = self._build_empty_dict(keys_flat)
    #     init_dict = self._fill_dict(empty_dict, mode=mode)
    #     return init_dict

    # def _build_empty_dict(self, keys_flat):
    #     """
    #     Accepts a dictionary of lists and constructs an empty dictionary according
    #     to that structure.
    #     """
    #     # Build each of the individual dictionaries.
    #     sub_dicts = []
    #     dict_names = []
    #     for i,k in enumerate(keys_flat):
    #         sub_dicts.append({})
    #         for k1 in keys_flat[k]:
    #             sub_dicts[i][k1] = None
    #         dict_names.append(k)
    #     # Assign dictionaries to appropriate value of parent dictionary.
    #     for sub in sub_dicts:
    #         for k in sub:
    #             if len(k) > 6:
    #                 if k[:6] == 'times_':
    #                     sub_name = k[6:]
    #                     sub_ind = dict_names.index(sub_name)
    #                     sub[k] = sub_dicts[sub_ind]
    #     empty_dict = sub_dicts[dict_names.index('_main')]
    #     return empty_dict

    # def _fill_dict(self, dict_to_fill, mode='single'):
    #     for k,v in dict_to_fill.iteritems():
    #         if isinstance(v, dict):
    #             dict_to_fill[k] = self._fill_dict(v, mode=mode)
    #         else:
    #             if mode=='multi-shared':
    #                 dict_to_fill[k] = [mp.RawValue('d') for _ in xrange(self.n_proc)]
    #             elif mode=='multi':
    #                 dict_to_fill[k] = [0.] * self.n_proc
    #             elif mode=='single':
    #                 dict_to_fill[k] = 0.
    #             else:
    #                 dict_to_fill[k] = 0.
    #      return dict_to_fill

    # def _accum_stamps(self, times_dict, stamp_keys, timestamps):
    #     for k,v in stamp_keys.iteritems():
    #         if isinstance(v, dict):
    #             self._accum_stamps(times_dict[k], v, timestamps)
    #         else:
    #             times_dict[k] += timestamps[v[0]] - timestamps[v[1]]

    # def _write_to_shared_dict(self, rank, shared_dict, private_dict):
    #     """
    #     shared_dict and private_dict must have the same key structure,
    #     shared_dict must contain lists of *values* only (possibly in nested dicts)
    #     """
    #     for k in private_dict:
    #         if isinstance(private_dict[k], dict):
    #             self._write_to_shared_dict(rank, shared_dict[k], private_dict[k])
    #         else:
    #             shared_dict[k][rank].value = private_dict[k]

        # self._accum_stamps(times_master, stamp_keys, timestamps)

        # for times_name, times_dict in times_dicts.iteritems():
        #     if times_dict is not None:
        #         self.collect_time_dict(times_master[times_name], times_dict)

    # def collect_time_dict(self, master_dict, recent_dict):
    #     """
    #     master_dict and recent_dict must have the same key structure.
    #     """
    #     for k,v in recent_dict.iteritems():
    #         if isinstance(v, dict):
    #             self.collect_time_dict(master_dict[k], v)
    #         else:
    #             master_dict[k] += v