from rllab import config
import subprocess
import os
import contextlib


@contextlib.contextmanager
def using_tmp_file(content):
    import tempfile
    f = tempfile.NamedTemporaryFile()
    try:
        f.write(content)
        f.flush()
        yield f
    finally:
        f.close()


class ResourceManager(object):
    def __init__(
            self,
            s3_resource_path,
            local_resource_path,
    ):
        self.s3_resource_path = s3_resource_path
        self.local_resource_path = local_resource_path

    def register_data(self, resource_name, content):
        with using_tmp_file(content) as f:
            self._upload(resource_name, f.name)
            self._upload_local(resource_name, f.name)

    def register_file(self, resource_name, file_name):
        self._upload(resource_name, file_name)
        self._upload_local(resource_name, file_name)

    def _upload(self, resource_name, file_name):
        subprocess.check_call([
            "aws",
            "s3",
            "cp",
            file_name,
            os.path.join(self.s3_resource_path, resource_name)
        ])

    def _upload_local(self, resource_name, file_name):
        local_file_name = os.path.join(self.local_resource_path, resource_name)
        local_dir_name = os.path.dirname(os.path.realpath(local_file_name))

        subprocess.check_call([
            "mkdir",
            "-p",
            local_dir_name
        ])
        subprocess.check_call([
            "cp",
            file_name,
            local_file_name,
        ])

    def _upload_content(self, resource_name, content):
        with using_tmp_file(content) as f:
            self._upload(resource_name, f.name)

    def get_file(self, resource_name, mkfile=None):
        local_file_name = os.path.join(self.local_resource_path, resource_name)
        s3_file_name = os.path.join(self.s3_resource_path, resource_name)
        if os.path.exists(local_file_name):
            return local_file_name
        try:
            subprocess.check_call([
                "aws",
                "s3",
                "cp",
                s3_file_name,
                local_file_name,
            ])
            if os.path.exists(local_file_name):
                return local_file_name
        except Exception as e:
            print(e)
        if mkfile is not None:
            mkfile()
            return self.get_file(resource_name)
        raise FileNotFoundError()


resource_manager = ResourceManager(
    s3_resource_path=config.AWS_S3_RESOURCE_PATH,
    local_resource_path=os.path.join(config.PROJECT_PATH, "data/resource")
)
