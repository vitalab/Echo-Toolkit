from matplotlib import pyplot as plt
import matplotlib.animation as animation


def show_gif(sequence, title=None):
    fig, ax = plt.subplots()
    if title:
        fig.suptitle(title)
    im = ax.imshow(sequence[0], animated=True)

    def update(i):
        im.set_array(sequence[i])
        return im,

    animation_fig = animation.FuncAnimation(fig, update, frames=len(sequence), interval=75, blit=True, repeat_delay=10,)
    return animation_fig
