# KT-Net
KT-Net: Knowledge Transfer for Unpaired 3D Shape Completion

conda create -n kt-net python=3.8
conda activate kt-net


cd net/util/emd_module
python setup.py install
cd ../../..

sh run_KT.sh 