set -x
echo "----------------------------------------------------------------------------------------------"
date
sudo chown -R lpatel.lpatel ~/projects/repos/xgboost/

cd /home/lpatel/projects/repos/xgboost/build
cmake .. 
make -j 4 

cd /home/lpatel/projects/repos/xgboost/python-package 
sudo /usr/bin/pip3 uninstall -y xgboost   || true
sudo /usr/bin/python3 setup.py install #--no-cache-dir

#/usr/bin/python3 /home/lpatel/projects/repos/xgboost/z_xgboost_example.py
/usr/bin/python3 /home/lpatel/projects/repos/xgboost/z_xgboost_aki_tesing.py
