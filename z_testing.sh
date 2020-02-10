set -x
echo "----------------------------------------------------------------------------------------------"
date

cd /home/lpatel/projects/repos/xgboost/build
cmake .. 
make -j 4 

cd /home/lpatel/projects/repos/xgboost/python-package 
sudo pip3 uninstall -y xgboost   || true
sudo python3 setup.py install 

python3 xgboost_example.py
