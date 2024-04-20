import json
import matplotlib.pyplot as plt

numoflayers = 1
numofnodes = [1000,500,30]
fold = 4
data = []
for i in range(len(numofnodes)):
    with open(f"convergence/points{numoflayers}{numofnodes[i]}{fold}.json", "r") as json_file:
        data.append(json.load(json_file))

for i in range(len(data)):
    plt.plot(data[i], label=f'nodes={numofnodes[i]}')

plt.legend(loc='upper right', fontsize='large', shadow=True, fancybox=True)

plt.xlabel("epoch")
plt.ylabel("loss")
plt.title(f"Convergence of RMSE, Fold={fold}, Layers={numoflayers}.")
plt.grid(True)
# plt.show()
plt.savefig(f'convergence_plot/fold{fold}.png')

