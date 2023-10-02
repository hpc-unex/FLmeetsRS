import os

cd patch
cp -r yamls ../FedML/python/
cp -r main.py ../FedML/python/
cp -r launchMPI.sh ../FedML/python/
cp -r algorithms ../FedML/python/
cp -r data_loader.py ../FedML/python/fedml/data/
cp -r arguments.py ../FedML/fedml/
cp -r __init__py ../FedML/fedml/
cp -r RSICB256 ../FedML/python/fedml/data
mkdir ~/FLmeetsRS/FedML/fedml_data/RSICB256

cd ../FedML
pip install fedml
