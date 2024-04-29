import json
import matplotlib.pyplot as plt

numoflayers = [1,2,3]#1
numofnodes = 30#[800,100,30]
folds = 5
for f in range(0,folds):
    data = []
    for i in range(len(numoflayers)):
    # for i in range(len(numofnodes)):
        # with open(f"convergence/points{numoflayers}{numofnodes[i]}{f}.json", "r") as json_file:
        with open(f"convergence/points{numoflayers[i]}{numofnodes}{f}.json", "r") as json_file:
            data.append(json.load(json_file))

    for d in range(len(data)):
        # plt.plot(data[d], label=f'nodes={numofnodes[d]}')
        plt.plot(data[d], label=f'layers={numoflayers[d]}')

    plt.legend(loc='upper right', fontsize='large', shadow=True, fancybox=True)

    plt.xlabel("epoch")
    plt.ylabel("loss")
    # plt.title(f"Convergence of RMSE, Fold={f}, Layers={numoflayers}.")
    plt.title(f"Convergence of RMSE, Fold={f}, Nodes={numofnodes}.")
    plt.grid(True)
    # plt.show()
    plt.savefig(f'convergence_plot/fold{f}.png')
    plt.clf()

