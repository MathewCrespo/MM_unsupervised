import os
for i in range(5):
    os.system('python main.py --log_root FM4-44_{} --test_fold {} --epoch 200 --batchsize 128 --base MM'.format(i+1,i+1))