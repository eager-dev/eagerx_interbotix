from matplotlib import animation
import matplotlib.pyplot as plt


def save_frames_as_gif(dt, frames, path='.', filename='swimm_animation.gif', dpi=15):
    """

    :param dt: 1/rate of the environment
    :param frames: list containing all the rgb arrays
    :param path: Path to the log directory where the gif is going to be saved
    :param filename: name of the gif e.g. "NAME_eps_1.gif"
    :param dpi: 72 dpi is ok.
    :return:
    """

    # Mess with this to change frame size
    fig = plt.figure(figsize=(frames[0].shape[1] / 72, frames[0].shape[0] / 72.0), dpi=dpi)
    ax = fig.gca()
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save('%s/%s' % (path, filename), writer='Pillow', fps=int(1 / dt))
    plt.close(fig)
    print('Gif saved to %s/%s' % (path, filename))
