#coding=utf-8

'''
@author LiangYu
@email  liangyufz@gmail.com
@create date 2018-07-13 18:55:11 
@modify date 2018-08-15 11:01:34
@desc [description]
'''

import numpy as np

def plot_hand(coords_hw, axis, prob=None, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
            if prob is not None:
                axis.text(coords_hw[connection[0], :][1], coords_hw[connection[0], :][0], '{a}:{b:.3f}'.format(a=connection[0], b=prob[connection[0]]))
                axis.text(coords_hw[connection[1], :][1], coords_hw[connection[1], :][0], '{a}:{b:.3f}'.format(a=connection[1], b=prob[connection[1]]))
            else:
                axis.text(coord1[1], coord1[0], str(connection[0]))
                axis.text(coord2[1], coord2[0], str(connection[1]))
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)
            axis.text(coord1[1], coord1[0], str(connection[0]))
    if prob is not None:
        axis.invert_yaxis()


def plot_prob(prob, axis):
    x = list(range(21))
    y = prob
    axis.plot(x, y, color='r', markerfacecolor='blue', marker='o')
    for a, b in zip(x, y):
        if b < 0.5:
            axis.text(a, b, "({a},{b:.3f})".format(a=a, b=b), ha='center', va='bottom', fontsize=10)
            