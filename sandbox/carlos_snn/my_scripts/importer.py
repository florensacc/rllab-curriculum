import joblib
import sys
import os.path as osp
import csv
import json
import imp

def load_problem(log_dir, iteration=None, pkl_file=None, json_file=None):
    # load pkl file
    if pkl_file is None:
        if iteration is None:
            pkl_file_path = osp.join(log_dir, "params.pkl")
        else:
            pkl_file_path = osp.join(log_dir, "itr_{}.pkl".format(iteration))
    else:
        pkl_file_path = "%s/%s" % (log_dir, pkl_file)

    if osp.isfile(pkl_file_path):
        data = joblib.load(pkl_file_path)  # load the pkl file
    else:
        print("Cannot find %s" % (pkl_file_path))
        sys.exit(1)
    # load json file
    if json_file is None:
        if iteration is None:
            json_file_path = osp.join(log_dir, "params.json")
        else:
            json_file_path = osp.join(log_dir, "itr_{}.json".format(iteration))
    else:
        json_file_path = "%s/%s" % (log_dir, json_file)

    if osp.isfile(json_file_path):
        params = json.load(open(json_file_path, "r"))  # load the json file
    else:
        print("Cannot find %s" % (json_file_path))
        sys.exit(1)

    # read algo (a bit cumbersome) ------------------------
    # algo_spec = params["json_args"]["algo"]
    # del algo_spec['hallucinator'], algo_spec['latent_regressor']
    # _name = algo_spec['_name']  # this is always the dot path of the module!
    # script_name = _name.split('.')[-2]
    # class_name = _name.split('.')[-1]
    # print('\n\n' + script_name + '\n\n')
    # algo_class = imp.load_source('%s' % class_name, 'sandbox/carlos_snn/old_my_snn/%s.py' % script_name)

    # ALGO = getattr(algo_class, class_name)

    # algo = ALGO(env=data["env"], policy=data["policy"], baseline=data["baseline"],
                # hallucinator=data['algo'].hallucinator, latent_regressor=data['algo'].latent_regressor, **algo_spec)
    # data["algo"] = algo

    data["progress"] = read_csv("%s/progress.csv" % (log_dir))

    return data


# only reads float numbers...
def read_csv(csvfile):
    with open(csvfile) as f:
        reader = csv.DictReader(f)
        data = dict()
        for key in reader.fieldnames:
            data[key] = []
        for row in reader:
            for key in reader.fieldnames:
                value = row[key]
                data[key].append(float(value))
    return data
