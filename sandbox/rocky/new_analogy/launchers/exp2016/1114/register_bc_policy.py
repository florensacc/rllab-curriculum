from sandbox.rocky.s3.resource_manager import resource_manager
resource_name = "irl/claw-bc-pretrained-v1.pkl"
resource_manager.register_file(resource_name, file_name="params.pkl")