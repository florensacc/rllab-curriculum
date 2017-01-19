import os
import subprocess
import shutil
import datetime


def autoclone(file_path):
    print('trying to autoclone the file: ', file_path)
    # import pdb; pdb.set_trace()
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    dir_name = os.path.dirname(file_path)

    i = 0
    while file_name[i].isdigit():
        i += 1
        if file_name[i] in ['_', '-']:
            i += 1
    d = datetime.date.today()
    day_tag = d.strftime('%d-%m-')

    j = None
    v_num = 0
    if file_name[-3:-1] == '_v' and file_name[-1].isdigit():  # this only allows up to 10 versions!
        v_num = int(file_name[-1]) + 1
        j = -3
    v_tag = '_v' + str(v_num)

    new_file_name = day_tag + file_name[i:j] + v_tag + '.py'
    new_file_path = os.path.join(dir_name, new_file_name)

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
