import tempfile

from rllab import config
import subprocess
import os
import contextlib


@contextlib.contextmanager
def using_tmp_file(content):
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

    def register_data(self, resource_name, content, compress=False):
        with using_tmp_file(content) as f:
            self._upload(resource_name, f.name, compress=compress)
            self._upload_local(resource_name, f.name)

    def register_file(self, resource_name, file_name, compress=False):
        self._upload(resource_name, file_name, compress=compress)
        self._upload_local(resource_name, file_name)

    def _upload(self, resource_name, file_name, compress=False):
        if compress:
            # upload the compressed file
            compressed_resource_name = resource_name + ".gz"
            f = tempfile.NamedTemporaryFile(); f.close()
            file_dir = os.path.dirname(file_name)
            file_name_only = os.path.basename(file_name)
            cmd = [
                "tar",
                "-zcf",
                f.name,
                "-C",
                file_dir,
                file_name_only,
            ]
            # import ipdb; ipdb.set_trace()
            # print(cmd)
            subprocess.check_call(cmd)
            subprocess.check_call([
                "aws",
                "s3",
                "cp",
                f.name,
                os.path.join(self.s3_resource_path, compressed_resource_name)
            ])
        else:
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

    def get_file(self, resource_name, mkfile=None, compress=False):
        local_file_name = os.path.join(self.local_resource_path, resource_name)
        s3_file_name = os.path.join(self.s3_resource_path, resource_name)
        if os.path.exists(local_file_name):
            return local_file_name
        if compress:
            try:
                s3_file_name = s3_file_name + ".gz"
                f = tempfile.NamedTemporaryFile(); f.close()
                local_compressed_file_name = f.name

                subprocess.check_call([
                    "aws",
                    "s3",
                    "cp",
                    s3_file_name,
                    local_compressed_file_name,
                ])
                f = tempfile.NamedTemporaryFile(); f.close()
                decompress_folder_name = f.name
                cmd = [
                    "mkdir",
                    "-p",
                    decompress_folder_name
                ]
                print(cmd)
                subprocess.check_call(cmd)
                cmd = [
                    "tar",
                    "-zxf",
                    local_compressed_file_name,
                    "-C",
                    decompress_folder_name
                ]
                subprocess.check_call(cmd)
                decompressed_file = os.listdir(decompress_folder_name)[0]
                local_decompressed_file_name = os.path.join(decompress_folder_name, decompressed_file)
                if os.path.exists(local_decompressed_file_name):
                    subprocess.check_call([
                        "cp",
                        local_decompressed_file_name,
                        local_file_name
                    ])
                    return local_file_name
            except Exception as e:
                print(e)
        else:
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
            return self.get_file(resource_name, compress=compress)
        raise FileNotFoundError()


resource_manager = ResourceManager(
    s3_resource_path=config.AWS_S3_RESOURCE_PATH,
    local_resource_path=os.path.join(config.PROJECT_PATH, "data/resource")
)
