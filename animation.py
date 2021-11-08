import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d, Axes3D


def create_animation_1d(parameters, probabilities, file_name):
    """
    1自由度のアニメーションを作成するメインの関数

    See Also
    --------
    framesの数だけ画像を合成する
    animation_plotに値を渡すためにクロージャにしている（おそらく）
    """
    fig = plt.figure()

    x_min = parameters[0]
    x_max = parameters[-1]

    frames = len(probabilities)

    def animation_plot(frame):
        """
        FuncAnimationクラスに渡すフレームを作成する関数

        See Also
        --------
        plt.cla()で毎回軸をクリアする必要がある
        """
        plt.cla()  # 軸をクリアする
        plt.plot(parameters, probabilities[frame])  # プロット
        plt.xlim([x_min, x_max])
        plt.ylim([0, 1])
        plt.title(f'Time: {(frame+1)*0.01:.2f}[s]')
        plt.ylabel('probability')

    # アニメーションを作成する。
    animation = FuncAnimation(fig, animation_plot, frames=frames)
    animation.save(f'gif/{file_name}.gif')

def create_animation_2d(k1_list, k2_list, probabilities, file_name):
    """
    2自由度のアニメーションを作成するメインの関数

    See Also
    --------
    framesの数だけ画像を合成する
    animation_plotに値を渡すためにクロージャにしている（おそらく）
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x_min = k1_list[0]
    x_max = k1_list[-1]
    y_min = k2_list[0]
    y_max = k2_list[-1]

    X, Y = np.meshgrid(k1_list, k2_list)

    frames = len(probabilities)

    def animation_plot(frame):
        """
        FuncAnimationクラスに渡すフレームを作成する関数

        See Also
        --------
        plt.cla()で毎回軸をクリアする必要がある
        """
        ax.cla()
        ax.plot_surface(X, Y, np.array(probabilities[frame]).reshape([len(k1_list), len(k2_list)]), cmap='seismic')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([0, 1])
        ax.set_title(f'Time: {(frame+1)*0.01:.2f}[s]')
        ax.set_xlabel('k1')
        ax.set_ylabel('k2')
        ax.set_zlabel('probability')

    # アニメーションを作成する。
    animation = FuncAnimation(fig, animation_plot, frames=frames)
    animation.save(f'gif/{file_name}.gif')