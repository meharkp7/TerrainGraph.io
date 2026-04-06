import matplotlib.pyplot as plt

def create_bar_chart(class_dist):
    names = list(class_dist.keys())
    values = list(class_dist.values())

    fig, ax = plt.subplots()
    ax.barh(names, values)
    ax.set_title("Terrain Distribution")

    return fig