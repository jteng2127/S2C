# in S2C/

conda create -n S2C python=3.8
conda activate S2C
conda config --env --add channels conda-forge

conda install pytorch=1.8.2 torchvision=0.9.2 torchaudio=0.8.2 cudatoolkit=11.1 -c pytorch-lts -c nvidia
# numpy np.bool is removed in 1.24
conda install python-dotenv tqdm matplotlib tensorboard scipy scikit-image numpy=1.23.5

# [linux - 'GLIBCXX_3.4.30' not found for librosa in conda virtual environment (after trying out a lot of solutions)? - Stack Overflow](https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin)
# cudnn for pytorch, nccl for mxnet
conda install -c conda-forge gcc=14.2.0 cudnn nccl

# [torch-scatter · PyPI](https://pypi.org/project/torch-scatter/)
conda install -c pyg pytorch-scatter 

# [Conda install incompatible with opencv for python 3.9 · Issue #3207 · pytorch/vision](https://github.com/pytorch/vision/issues/3207)
pip install opencv-contrib-python 

# Conda solving environment takes too long
pip install mxnet-cu111

# Segment Anything (with modifications in S2C)
CWD=$(pwd)
git clone https://github.com/facebookresearch/segment-anything.git ../S2C-segment-anything
cd ../S2C-segment-anything
cp $CWD/modeling/mask_decoder.py ./segment_anything/modeling/mask_decoder.py
cp $CWD/modeling/sam.py ./segment_anything/modeling/sam.py
pip install -e .
cd $CWD