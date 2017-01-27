from sandbox.rocky.s3.resource_manager import resource_manager
resource_name = "irl/I1-3k-bc-pretrained.pkl"
resource_manager.register_file(resource_name, file_name="params.pkl")