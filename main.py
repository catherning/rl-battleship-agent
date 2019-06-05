from General import General
from battleship.Field import Field as field
from battleship.nnet import nnet as nn


if __name__=="__main__":
    f = field(6) # init battlefield
    nnet = nn(f) # init NN

    # define where to store\get checkpoint and model
    checkpoint = './temp/'
    load_model = False
    models_folder = './models/'
    best_model_file = 'best.pth.tar'
    
    if load_model:
        nnet.load_checkpoint(models_folder, best_model_file)

    g = General(f, nnet) # init play and players
    
    if load_model:
        print("Load trainExamples from file")
        g.loadTrainExamples()
        
    g.fight()
