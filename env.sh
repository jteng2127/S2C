# in S2C/

set -euxo pipefail

eval "$(conda shell.bash hook)"

conda create -n S2C python=3.8.18
conda activate S2C
conda config --env --add channels conda-forge

# numpy np.bool is removed in 1.24
# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 numpy=1.22.3 -c nvidia -c pytorch -c conda-forge
conda install cudatoolkit=11.1 -c nvidia
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# conda install python-dotenv tqdm matplotlib tensorboard scipy scikit-image pandas fastai::opencv-python-headless=4.10.0.84 pyg::pytorch-scatter=2.0.8

pip install python-dotenv tqdm matplotlib tensorboard==2.12.0 scipy scikit-image pandas opencv-python-headless numpy==1.23.5 pillow==9.5.0
pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html
conda install pydensecrf numpy=1.23.5

# (Optional) for debugging
pip install debugpy

# Segment Anything (with modifications in S2C)
CWD=$(pwd)
git clone https://github.com/facebookresearch/segment-anything.git ../S2C-segment-anything
cd ../S2C-segment-anything
cp $CWD/modeling/mask_decoder.py ./segment_anything/modeling/mask_decoder.py
cp $CWD/modeling/sam.py ./segment_anything/modeling/sam.py
pip install -e .
cd $CWD

# # [linux - 'GLIBCXX_3.4.30' not found for librosa in conda virtual environment (after trying out a lot of solutions)? - Stack Overflow](https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin)
# # cudnn for pytorch, nccl for mxnet
# conda install -c conda-forge gcc=14.2.0 cudnn nccl

# # [torch-scatter · PyPI](https://pypi.org/project/torch-scatter/)
# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.2+cu111.html

# # [Conda install incompatible with opencv for python 3.9 · Issue #3207 · pytorch/vision](https://github.com/pytorch/vision/issues/3207)
# conda install 
# # pip install opencv-contrib-python 

# # Conda solving environment takes too long
# pip install mxnet-cu111
