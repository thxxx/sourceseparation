apt-get -y update
apt-get install -y ffmpeg
apt-get install -y libsndfile1-dev
git config --global user.email zxcv05999@naver.com
git config --global user.name thxxx
python -m pip install --upgrade pip
pip install diffusers transformers einops audiotools datasets
pip install .
pip install omegaconf jiwer pyroomacoustics jaxtyping
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install -U git+https://HL7644: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@github.com/OptimizerAI/data_downloader.git
huggingface-cli login
# mkdir pretrained/
# python download_weights.py
# bash custom/setup.sh
# wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt?download=true
# wget https://huggingface.co/Daniel777/melody/resolve/main/total_model_350.pth?download=true
# wget https://huggingface.co/Daniel777/melody/resolve/main/vae_weight.pth?download=true