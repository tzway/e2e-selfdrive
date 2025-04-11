Open Anaconda Prompt in the root folder of the project and run these commands

1) Create an environment & activate
conda create -n sdcar python=3.7 -y
2) Activate the environement
conda activate sdcar
3) Install the requirements
pip install -r requirements.txt 
4) Install additionnal libraries
pip install python-engineio==3.13.2
pip install python-socketio==4.6.1
5) Test
python drive.py
6)Now open your simulator in Audacity & run yes!!