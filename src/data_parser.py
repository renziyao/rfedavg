import matplotlib.pyplot as plt
import os
import argparse

def parse_acc(f):
    lines = f.readlines()
    source_acc = []
    target_acc = []
    for line in lines:
        if 'acc' in line:
            acc = float(line[12: -1])
            if 'source' in line:
                source_acc.append(acc)
            if 'target' in line:
                target_acc.append(acc)
    return source_acc, target_acc

def parse(fn):
    with open(fn, 'r') as f:
        return parse_acc(f)

def parse_and_plot(flist):
    for f in flist:
        _, tmp = parse(f)
        plt.plot(tmp, label=f)
    plt.legend()
    plt.show()

def plot_folder(rootdir):
    list = os.listdir(rootdir)
    flist = []
    for i in range(0,len(list)):
        if list[i][-3:] != 'txt': continue
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path):
            flist.append(path)
    parse_and_plot(flist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir', 
        type=str, 
        help='data directory',
    )
    args = parser.parse_args()
    plot_folder(args.dir)
