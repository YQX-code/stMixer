import argparse

class Stage1_Options():

    def __init__(self):
        self.parser=argparse.ArgumentParser(description='All needed arguments in stage1')
        self.parser.add_argument('--loss',default='ce',help='The loss function used for stage1 training')
        self.parser.add_argument('--learningRate',default='0.001',help='The learning rate used for stage1 training',type=float)
        self.parser.add_argument('--batchSize',default='3484',help='The batch size used for stage1 training simulated1296 real9196',type=int)#模拟数据1296 小鼠大脑9196 新数据20125 spleen1 2653 spleen2 2768  real3484
        self.parser.add_argument('--model',default='sc',help='The model used for stage1 training',type=str)
        self.parser.add_argument('--continueTrain',action='store_true',default=False,help='If True,load the existing model as init parameters for continue training')
        self.parser.add_argument('--trainPercentage',default='0.7',help='Percentage of training set occupies in the whole 2d dataset',type=float)
        self.parser.add_argument('--gpuID',default='cpu',help='Select the gpu(s) used for training',type=str)
        self.parser.add_argument('--epoch',default=200,help='Epoch number for training',type=int)
        self.parser.add_argument('--initAlgorithm',default='kaiming',help='The algorithm to initialize the network, not used in continue train')
        self.parser.add_argument('--testFrequency',default=1,help='The frequency of test after x epoches of training',type=int)
        self.parser.add_argument('--expName',default='1',help='The experimental name to save everything in the specific folder e.g.:stage1_model_save_path/expName',required=True)
        self.parser.add_argument('--continueExpName',default='',help='To continue train the saved model, specify the folder name of the experiment, only needed when continueTrain is True')
        self.parser.add_argument('--continueSavedModelName',default='',help='To continue trian the saved model, specify the .pth file named by the model, only needed when continuTrain is True')
        self.parser.add_argument('--continueTrainStartEpoch',default=1,help='define the start epoch for continue training',type=int)
        self.parser.add_argument('--seed', default=10,
                                 help='define the start epoch for continue training', type=int)
    def get_parser(self):
        return self.parser
