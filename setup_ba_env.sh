conda create -n ba -c conda-forge python=3.8
conda activate ba
python3 -m pip install git+https://github.com/centreborelli/sat-bundleadjust
pip install fire
conda deactivate
echo "ba conda env created !"
