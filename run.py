import os

## for auto run:
'''
for i in range(4,5):
    os.system('python main.py --log_root base{} --test_fold {} --epoch 200 --batchsize 128 --base base'.format(i+1,i+1))
'''

for i in range(2,6):
    os.system('CUDA_VISIBLE_DEVICES=7 python lincls_main.py --log_root imgnet_finetune{} --test_fold {} --pretrained base4 --batchsize 64 --imgnet True'.format(i,i))