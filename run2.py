import os
for i in range(5):
    os.system('python main2.py --log_root p3FM34-44_{} --test_fold {} --epoch 200 --batchsize 128 --base MM --gpu "4,5,6,7" --r 3 --fl 34'.format(i+1,i+1))