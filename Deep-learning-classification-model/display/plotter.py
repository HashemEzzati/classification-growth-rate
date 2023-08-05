from bokeh.plotting import figure, show
from bokeh.layouts import column
import numpy as np


def plotter(data, labels):
    fig_list = np.array([])
    for cls in labels:
        fig = figure(title='Prediction of class_( ' + str(cls) + ' )',
                     tools=['pan', 'box_zoom', 'ywheel_zoom', 'xwheel_zoom', 'reset'],
                     width=1200,
                     height=400)
        fig.toolbar.logo = None
        fig.xaxis.axis_label = 'Classes'
        fig.yaxis.axis_label = 'The number of class prediction'

        fig.vbar(labels, 0.5, 0, data[np.where(labels == cls)[0][0]], legend_label='class' + str(cls))

        fig_list = np.append(fig_list, fig)

    show(column(list(fig_list)))
