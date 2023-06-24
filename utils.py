def label_dict(array):
    d = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0}
    for y in array:
        if y == 0:
            d["0"] += 1
        elif y == 1:
            d["1"] += 1
        elif y == 2:
            d["2"] += 1
        elif y == 3:
            d["3"] += 1
        elif y == 4:
            d["4"] += 1
        elif y == 5:
            d["5"] += 1
        elif y == 6:
            d["6"] += 1
        elif y == 7:
            d["7"] += 1
        elif y == 8:
            d["8"] += 1
        elif y == 9:
            d["9"] += 1
        elif y == 10:
            d["10"] += 1
    return d


def print_dict(d):
    for k, v in d.items():
        # print(k, v)
        print(v)


import matplotlib.pyplot as plt


def pie_chart_lable(dictionary):
    # Data to plot
    labels = []
    sizes = []
    for x, y in dictionary.items():
        labels.append(x)
        sizes.append(y)
    # Plot
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()
