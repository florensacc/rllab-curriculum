import os
import subprocess
import datetime


def autoclone(file_path, launch_args):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    dir_path = os.path.dirname(file_path)

    # remove possible date at the start of the file name
    i = 0
    while file_name[i].isdigit():
        i += 1
        if file_name[i] in ['_', '-']:
            i += 1
    d = datetime.datetime.now()
    day_tag = d.strftime('%d-%m-')

    # remove possible version number at the end of file name
    j = None
    v_num = 0
    if file_name[-3:-1] == '_v' and file_name[-1].isdigit():  # this only allows up to 10 versions!
        v_num = int(file_name[-1]) + 1
        j = -3
    v_tag = '_v' + str(v_num)

    # construct new file name, checking that not taken
    file_core_name = file_name[i:j] if not launch_args.name else launch_args.name
    new_file_name = day_tag + file_core_name + v_tag + '.py'
    files = os.listdir(dir_path)
    while new_file_name in files:
        v_num += 1
        v_tag = '_v' + str(v_num)
        new_file_name = new_file_name = day_tag + file_name[i:j] + v_tag + '.py'
    new_file_path = os.path.join(dir_path, new_file_name)

    # insert the new docstring
    content = []
    with open(file_path, 'r') as f:
        for line in f:
            content.append(line)

    if '"""' not in content[0]:
        content.insert(0, '"""\n')
        content.insert(0, '"""\n')
    content.insert(1, d.strftime('%c') + ': ' + v_tag + '\n')

    with open(new_file_path, 'w') as f:
        for i in range(len(content)):
            f.write(content[i])

    # git checkout current file and commit new one
    subprocess.call(["git", "reset"])
    subprocess.call(["git", "checkout", file_path])
    subprocess.call(["git", "add", new_file_path])
    subprocess.call(["git", "commit", '-m', new_file_name])


