import os

## for auto run:
for i in range(4):
    os.system('python main.py --log_root base{} --test_fold {} --epoch 200 --batchsize 128 --base base'.format(i+1,i+1))