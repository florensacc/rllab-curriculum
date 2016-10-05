import logging
import os
import re

logger = logging.getLogger(__name__)

_HOST_MD_FIELD = 'hostname'


def get_self_host():
    try:
        with open('/etc/chef/metadata.txt') as metadata_f:
            for line in metadata_f:
                if re.match(r'^{}=[\w.]+'.format(_HOST_MD_FIELD), line):
                    return line[len(_HOST_MD_FIELD)+1:].strip()
    except IOError:
        logger.warn('Failed to read metadata')

    logger.warn('Failed to find host from metadata')
    return None


def get_user():
    return os.environ.get('OPENAI_USER') or os.environ.get('USER')
