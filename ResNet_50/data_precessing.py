from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import shutil

train_filenames = os.listdir('') # 放train的路徑
train_cat = list(filter(lambda x:x[:3] == 'cat', train_filenames))
train_dog = list(filter(lambda x:x[:3] == 'dog', train_filenames))
x = ['train_cat', 'train_dog', 'test']
y = [len(train_cat), len(train_dog), len(os.listdir('test'))]

mytrain, myvalid = train_test_split(train_filenames, test_size=0.1)
mytrain, mytest = train_test_split(mytrain, test_size = 1/9)

mytrain_cat = list(filter(lambda x:x[:3] == 'cat', mytrain))
mytrain_dog = list(filter(lambda x:x[:3] == 'dog', mytrain))
myvalid_cat = list(filter(lambda x:x[:3] == 'cat', myvalid))
myvalid_dog = list(filter(lambda x:x[:3] == 'dog', myvalid))
mytest_cat = list(filter(lambda x:x[:3] == 'cat', mytest))
mytest_dog = list(filter(lambda x:x[:3] == 'dog', mytest))
x = ['mytrain_cat', 'mytrain_dog', 'myvalid_cat', 'myvalid_dog',  'mytest_cat', 'mytest_dog']
y = [len(mytrain_cat), len(mytrain_dog), len(myvalid_cat), len(myvalid_dog), len(mytest_cat), len(mytest_dog)]


def remove_and_create_class(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    os.mkdir(dirname+'/cat')
    os.mkdir(dirname+'/dog')

remove_and_create_class('mytrain')
remove_and_create_class('myvalid')
remove_and_create_class('mytest')

for filename in mytrain_cat:
    os.symlink('../../train/'+filename, 'mytrain/cat/'+filename)

for filename in mytrain_dog:
    os.symlink('../../train/'+filename, 'mytrain/dog/'+filename)

for filename in myvalid_cat:
    os.symlink('../../train/'+filename, 'myvalid/cat/'+filename)

for filename in myvalid_dog:
    os.symlink('../../train/'+filename, 'myvalid/dog/'+filename)

for filename in mytest_cat:
    os.symlink('../../train/'+filename, 'mytest/cat/'+filename)

for filename in mytest_dog:
    os.symlink('../../train/'+filename, 'mytest/dog/'+filename)