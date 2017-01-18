def get_time_stamp():
    import datetime
    import dateutil.tz
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S_%f')
    return timestamp

def create_dir_if_needed(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
