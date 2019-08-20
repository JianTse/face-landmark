#!/usr/bin/python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def is_number(s):
    try:
        float(s)  # for int, long and float
    except ValueError:
        try:
            complex(s)  # for complex
        except ValueError:
            return False
    return True

#log_file = sys.argv[1]
log_file = '../model/log.txt'
lines = open(log_file, 'r').readlines()

train_loss = []
test = [[] for _ in range(2)]
test_acc=[]
iter,iter_test = [],[]
i = 0
idx = [1,2]
while i < len(lines):
    line = lines[i]
    if 'solver.cpp:239] Iteration' in line:
        line = line.split()
        digs = line[5].split(',')
        iter.append(int(digs[0]))  # get iter num
        train_loss.append(float(line[-1]))  # get train loss
        i += 1
    elif 'Testing net (#0)' in line:
        line = line.split()
        #print(line)
        iter_test_count = int(line[5][:-1])
        iter_test.append(int(line[5][:-1]))
        #while 'Restarting data prefetching from start.' in lines[i]:
        #    i += 1
        '''
        for n,j in enumerate(idx):
            #print 'line', i+j, '---', lines[i+j].split()[-1]
            test[n].append(float(lines[i+j].split()[-1]))
        '''
        #i += 9
        lastParam = lines[i + 1].split()[-1]
        if is_number(lastParam) == True:
            test_acc.append(float(lastParam))
        else:
            lastParam = lines[i + 2].split()[-1]
            test_acc.append(float(lastParam))
        i += 1
    else:
        i += 1

# 画在一个图上面
plt.figure('train-graph')
plt.plot(iter,train_loss, label = 'loss')
plt.plot(iter_test,test_acc,label='acc')
plt.ylim([0,1])
plt.legend(loc='lower right')
plt.legend()

'''
# 画在不同的图上面
plt.figure('train')
plt.plot(iter,train_loss, label = 'loss')
plt.ylim([0,1])
plt.figure('test')
plt.plot(iter_test,test_acc,label='acc')
plt.legend(loc='lower right')
plt.legend()
'''

'''
for i, t in enumerate(test):
    #print 'iter_test', iter_test
    #print 't', t
    plt.plot(iter_test,t,label=labels[i])
plt.legend()

for i, t in enumerate(test):
    t = np.asarray(t,dtype=np.float32)
    print(labels[i], iter_test[t.argmax()], t.max())
'''
plt.show()
# print(iter)
# print(train_loss)
# print(iter_test)
# print(test)
