#!/usr/bin/env python
# from control3.common import *
# from control3.diagnostic_common import *
from collections import OrderedDict
from fnmatch import fnmatch
from glob import glob
from edison.tabulate import tabulate
import os.path as osp, itertools,numpy as np
from edison import (
    string2numlist,
    disp_dict_as_2d_array,disp_dict_as_3d_array,compute_mean_std_across_runs,load_hdfs_as_dataframe,
    concatenate,check_output_and_print
)
from edison.publication_plots import load_hdfs_as_dataframe
from pandas.stats.moments import ewma

def disp_info(df):
    stat_names = df.stat_name.unique()
    script_names = df.script_name.unique()
    statname2scriptnames = {stat_name:['']*len(script_names) for stat_name in stat_names}
    for (i_script,(script_name,script_grp)) in enumerate(sorted(df.groupby("script_name"))):
        for stat_name in script_grp.stat_name.unique():
            statname2scriptnames[stat_name][i_script] = 'x'
    print tabulate([[stat_name]+li for (stat_name,li) in statname2scriptnames.items()], headers = map(str,range(len(script_names))))
    print "--------------------"
    print "    Script Names    "
    for (i,script_name) in enumerate(script_names):
        print "%i: %s"%(i,script_name)


def filter_dataframe(df, col, desc):

    if "," in desc:
        scriptnames = desc.split(",")
        filtfn = lambda s: s in scriptnames
    elif "*" in desc:
        filtfn = lambda s: fnmatch(s, desc)
    else:
        filtfn = lambda s: s==desc
    
    vals = filter(filtfn, df[col].unique() )

    mask = [item in vals for item in df[col]]

    return df[mask]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--disp_info",action="store_true")
    parser.add_argument("--stat_names",help="comma-separated list of diagnostic stats you want to display",default="")    
    parser.add_argument("--avg_runs",action="store_true")
    parser.add_argument("--output",choices=["table","lc"],default="table")
    parser.add_argument("--table_style",choices=["0","1","2","3"],help="0: script/stat. 1: cfg+run/test+stat. 2: stat/cfg+run/test, 3: cfg+run/stat",default="0")
    parser.add_argument("--table_stat",choices=["final","mean"],default="final")
    parser.add_argument("--plot_mode",choices=["show","save"],default="show")
    parser.add_argument("--versions",action="store_true")
    parser.add_argument("--n_cfg",type=int)
    parser.add_argument("--label_lines_by",choices=["script","cfg","all"],default="script")
    parser.add_argument("--save_plot_to")
    parser.add_argument("--script_include")
    parser.add_argument("--cfg_include")
    parser.add_argument("--test_include")
    parser.add_argument("--smooth_span",type=int,default=0)
    args = parser.parse_args()

    if osp.isfile(args.dir):
        h5files = [args.dir]
    else:
        h5files = []
        for dirname in args.dir.split(","): 
            h5files.extend( glob(osp.join(dirname,"*.h5")) )
        if len(h5files) == 0:
            h5files = glob(osp.join(args.dir,"*/*.h5"))
    assert len(h5files) > 0

    assert args.disp_info or len(args.stat_names)>0

    if args.versions:
        gitrevlist = [line[:7] for line in check_output_and_print("cd $CTRL_ROOT && git rev-list HEAD").split("\n")]
        cfg_sortkey = lambda cfg: gitrevlist.index(cfg[:7])
        h5files = concatenate([glob(osp.join(args.dir, "%s/*.h5"%rev)) for rev in sorted(gitrevlist, key=cfg_sortkey)[:args.n_cfg]])

    df = load_hdfs_as_dataframe(h5files)

    for prefix in ["test","cfg","script"]:
        filter_pat = args.__dict__["%s_include"%prefix]
        if filter_pat:
            df = filter_dataframe(df, "%s_name"%prefix,filter_pat)


    if args.disp_info:
        disp_info(df)
    else:
        if args.stat_names == "all":
            stat_names=df.stat_name.unique()
        else:
            stat_names = string2numlist(args.stat_names,str)
        colors = 'bgrcmyk'
        styles = ['-x','-','--','-.']
        linestyles = [color+style for style in styles for color in colors ]

        if args.avg_runs:
            df = compute_mean_std_across_runs(df)

        if args.output == "lc":
            import matplotlib.pyplot as plt
            # Plotting 6D data! figure=test, axis=stat, cfg=linecolor, run=<nothing>, xaxis=time, yaxis=value
            for (test_name,test_name_grp) in df.groupby("test_name"):                     
                stat_names_here = list(set(test_name_grp.stat_name.unique()).intersection(stat_names))
                stat_names_here.sort(key = lambda x : stat_names.index(x))

                # axarr = [plt.figure().gca() for _ in xrange(len(stat_names_here))]
                _, axarr = plt.subplots(len(stat_names_here), sharex=True)                     
                if not isinstance(axarr,np.ndarray):
                    axarr = [axarr]
                # plt.title(test_name)           
                for (stat_name,stat_name_grp) in test_name_grp.groupby("stat_name"):
                    if stat_name not in stat_names: continue
                    i_stat = stat_names_here.index(stat_name)
                    axarr[i_stat].set_title("%s: %s"%(test_name,stat_name))
                    cfg2line = OrderedDict()
                    ax = axarr[i_stat]
                    if args.label_lines_by == "script":
                        it = stat_name_grp.groupby("script_name")
                    elif args.label_lines_by == "cfg":
                        it = stat_name_grp.groupby("cfg_name")
                    else:
                        it = stat_name_grp.groupby(["script_name","i_run"])
                    if args.versions: it = sorted(it, key=lambda (cfg_name,_) : cfg_sortkey(cfg_name))                    
                    if args.n_cfg: it = itertools.islice(it,0,args.n_cfg)
                    for (i_cfg,(cfg_name,cfg_name_group)) in enumerate(it):
                        linestyle = linestyles[i_cfg % len(linestyles)]
                        for timeseries in cfg_name_group["timeseries"]:
                            if args.smooth_span > 0: timeseries = ewma(timeseries,span=args.smooth_span)
                            line,=ax.plot(timeseries,linestyle)
                            if args.avg_runs:
                                assert len(cfg_name_group)==1
                                stderr = cfg_name_group["timeseries_stderr"].irow(0)
                                ax.fill_between(np.arange(len(timeseries)), timeseries-stderr,timeseries+stderr,color=linestyle[0],alpha=.2)
                            cfg2line[cfg_name]=line
                    ax.legend(cfg2line.values(),cfg2line.keys(),prop={'size':8}).draggable()
            if args.save_plot_to:
                plt.savefig(args.save_plot_to)
            else:
                plt.show()

        elif args.output == "table":
            data = {}


            cfg_run_repr = lambda cfg_name,i_run: cfg_name if args.avg_runs else "%s-RUN%i"%(cfg_name,i_run)
            if args.table_style == "0":
                keyfunc = lambda cfg_name,i_run,stat_name,test_name,script_name : (cfg_run_repr(script_name,i_run),stat_name)       
                table_dim = 2
            elif args.table_style == "1":
                keyfunc = lambda cfg_name,i_run,stat_name,test_name,script_name : (cfg_run_repr(cfg_name,i_run),"%s-%s"%(test_name,stat_name))
                table_dim = 2
            elif args.table_style == "2":
                table_dim = 3
                keyfunc = lambda cfg_name,i_run,stat_name,test_name,script_name : (stat_name,cfg_run_repr(cfg_name,i_run),test_name)
            elif args.table_style == "3":
                table_dim = 2
                keyfunc = lambda cfg_name,i_run,stat_name,test_name,script_name : (cfg_run_repr(cfg_name,i_run),stat_name)
            else:
                raise RuntimeError

            it = df.groupby(['cfg_name','i_run','stat_name','test_name','script_name'])
            for ((cfg_name,i_run,stat_name,test_name,script_name),grp) in it:
                if args.versions: cfg_name="%.4i-%s"%(cfg_sortkey(cfg_name),cfg_name)
                if stat_name in stat_names:
                    assert len(grp)==1
                    key = keyfunc(cfg_name,i_run,stat_name,test_name,script_name)
                    data[key] = (grp["timeseries"].irow(0).mean() if args.table_stat == "mean" else grp["timeseries"].irow(0)[-1])

            if table_dim == 2:
                disp_dict_as_2d_array(data,use_numeric_keys=True)
            elif table_dim == 3:
                disp_dict_as_3d_array(data,use_numeric_keys=True)


if __name__ == "__main__":
    main()