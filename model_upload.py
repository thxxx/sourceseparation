# 4000 80
# huggingface-cli login
from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="/home/khj6051/mel_con_sample/main/stable_audio_origin_weight.pth",
    path_in_repo="stable_audio_origin_weight.pth",
    repo_id="Daniel777/ss",
    repo_type="model"
)