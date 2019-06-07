from train_pytorch import General
from Field import Field
from nnet import CNNet, ResidualNNet

if __name__ == "__main__":

    network = "residual"

    f = Field(6)  # init battlefield

    if network == "residual":
        nnet = ResidualNNet(f)  # init NN
    elif network == "cnn":
        nnet = CNNet(f)

    # define where to store\get checkpoint and model
    checkpoint = './temp/'
    load_model = False
    models_folder = './models/'
    best_model_file = 'best.pth.tar'

    if load_model:
        nnet.load_checkpoint(models_folder, best_model_file)

    g = General(f, nnet)  # init play and players

    if load_model:
        print("Load trainExamples from file")
        g.load_train_examples()

    g.fight()
