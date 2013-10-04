
# python standard library
from collections import namedtuple
import time
import random
import multiprocessing
import os
import warnings

# third party
import matplotlib.pyplot as plt
import numpy
from scipy.stats import gaussian_kde



INCREMENT = 1


DataTuple = namedtuple('DataTuple', 'probability exponent range elapsed'.split())


def generate_flips(trials, collection_size=4):
    """
    Generates lists of random 0's or 1s

    :yield: count of random 1s generated in collection_size flips
    """
    for trial in xrange(trials):
        yield sum((random.randint(0,1) for flip in xrange(collection_size)))

def optimize_trials(max_exponent, desired_count=3,
                    thread_count=10, high_percentile=95,
                    low_percentile=5, tolerance=0.01):
    queue = multiprocessing.Queue()
    start_time = time.time()
    for exponent in xrange(1, max_exponent + 1):
        trials = 10**exponent
        threads = []
        for t in xrange(thread_count):
            thread = multiprocessing.Process(target=get_probability,
                                             kwargs={'trials':trials,
                                                     'desired_heads':desired_count,
                                                     'queue':queue})
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        data = []
        while not queue.empty():
            data.append(queue.get())
        percentile_range = (numpy.percentile(data, high_percentile)-
                            numpy.percentile(data, low_percentile))
        if percentile_range <= tolerance:
            return DataTuple(probability = numpy.median(data),
                             exponent=exponent,
                             range=percentile_range,
                             elapsed=time.time()-start_time)

    return DataTuple(probability = numpy.median(data),
                        exponent = exponent,
                        range = percentile_range,
                        elapsed = time.time()-start_time)


def get_probability(trials, desired_heads, queue=None):
    flips = generate_flips(trials=trials)
    if queue is not None:
        queue.put(sum(INCREMENT for count in flips if count==desired_heads)/float(trials))
        return
    return sum(INCREMENT for count in flips if count==desired_heads)/float(trials)


PLOT_FOLDER = 'figures'


class PlotSetup(object):
    '''
    A Context manager for plotting functions
    '''
    def __init__(self, label='', title='', xlabel='', ylabel='',
                 xlim=None, axe=None, figure=None,
                 legend_location='upper right',
                 plot_type='plot', file_type='png', output_folder=PLOT_FOLDER):
        """
        SetupPlot constructor

        :param:

         - `title`: title for the plot
         - `label`: string used to create output filename
         - `xlabel`: label for the x-axis
         - `ylabel`: label for the y-axis
         - `xlim`: min and max for the x-axis
         - `plot_type`: label for filename
         - `plot_folder`: sub-folder for output file
         - `axe`: existing axis to use instead of creating a new one
         - `figure`: existing figure to use
        """
        self.xlim = xlim
        self.title = title
        self.label = label
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_location = legend_location
        self.output_folder = output_folder
        self._path = None
        self._figure = figure
        self._axe = axe
        return

    @property
    def axe(self):
        """
        The current axis for plotting
        """
        if self._axe is None:
            # gca : get current axis (matplotlib is surprisingly unfriendly)
            self._axe = self.figure.gca()
        return self._axe

    @property
    def figure(self):
        """
        matplotlib.pyplot figure for the axis
        """
        if self._figure is None:
            self._figure = plt.figure()
        return self._figure

    @property
    def path(self):
        """
        Output path for the file
        """
        if self._path is None:
            filename = '{0}_{1}.png'.format(self.label, self.plot_type)
            self._path = os.path.join(PLOT_FOLDER, filename)
        return self._path

    
    def __enter__(self):
        """
        Does the common setup for the plotters

        :postcondition: figure, axe exist
        """
        self.axe.set_title(self.title)
        return self

    def __exit__(self, type, value, traceback):
        """
        creates the xlabel, legend, sets xlim and saves the figure to a file
        """        
        self.axe.set_xlabel(self.xlabel)
        self.axe.set_ylabel(self.ylabel)
        if self.xlim is not None:
            self.axe.set_xlim(self.xlim)

        # boxplot can't set labels so this will complain
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.axe.legend(loc=self.legend_location)
        self.figure.savefig(self.path)
        return



def plot_hline(y, plot_label, axe=None, figure=None, xlabel='', title='', xlim=None,
               legend_location='upper right',
               *args, **kwargs):
    """
    plot the hline

    :param:

     - `plot_label`: prefix for file_name
     - `y`: height of horizontal line
     - `xlabel`: label for the x-axis
     - `title`: title for the plot
     - `xlim`: min and max x-axis values

    :postcondition: figures/<label>_hline.png is a plot of data

    :return: PlotSetup
    """
    with PlotSetup(label=plot_label, title=title, axe=axe, figure=figure,
                   legend_location=legend_location, plot_type='hline') as plotter:
        plotter.axe.axhline(y, *args, **kwargs)
    return plotter


def plot_vline(x, plot_label, axe=None, figure=None, xlabel='', title='', xlim=None,
               legend_location='upper right',
               *args, **kwargs):
    """
    plot the hline

    :param:

     - `plot_label`: prefix for file_name
     - `x`: x-axis location of line
     - `xlabel`: label for the x-axis
     - `title`: title for the plot

    :postcondition: figures/<label>_vline.png is a plot of data

    :return: PlotSetup
    """
    with PlotSetup(label=plot_label, title=title,
                   legend_location=legend_location, plot_type='vline') as plotter:
        plotter.axe.axvline(x, *args, **kwargs)
    return plotter


def plot_histogram(data, plot_label, axe=None, figure=None,
                   xlabel='', title='', xlim=None,
                   legend_location='upper right',
                   *args, **kwargs):
    """
    plot the data on a histogram

    :param:

     - `plot_label`: prefix for file_name
     - `data`: collection of data
     - `xlabel`: label for the x-axis
     - `title`: title for the plot
     - `xlim`: min and max x-axis values

    :postcondition: figures/<label>_histogram.png is a plot of data

    :return: PlotSetup
    """
    if xlim is None:
        xlim = (min(0, min(data)), max(data))
    with PlotSetup(label=plot_label, title=title, axe=axe, figure=figure,
                   legend_location=legend_location, plot_type='histogram', xlabel=xlabel, xlim=xlim) as plotter:
        plotter.axe.hist(data, *args, **kwargs)
        plotter.axe.axvline(numpy.median(data), label='median', color='FireBrick')
        plotter.axe.axvline(numpy.mean(data), label='mean', color='DodgerBlue')
    return plotter


def plot_line(x_data, y_data, plot_label='', xlabel='', title='', xlim=None,
             ylabel='', axe=None, figure=None,
                   legend_location='upper right',
                   *args, **kwargs):
    """
    plot the data as a line

    :param:

     - `plot_label`: label for filename
     - `data`: collection of data
     - `xlabel`: label for the x-axis
     - `title`: title for the plot
     - `xlim`: min and max x-axis values

    :postcondition: figures/<label>_cdf.png is a plot of data

    :return: PlotSetup
    """
    with PlotSetup(label=plot_label, title=title, legend_location=legend_location, plot_type='line',
                   axe=axe, figure=figure,
                   xlabel=xlabel, xlim=xlim, ylabel=ylabel) as plotter:
        plotter.axe.plot(x_data, y_data,  *args, **kwargs)
    return plotter


def plot_cdf(data, plot_label='', xlabel='', title='', xlim=None,
             ylabel='', axe=None, figure=None,
                   legend_location='upper right',
                   *args, **kwargs):
    """
    plot the data as a Cumulative Distribuiton

    :param:

     - `plot_label`: label for filename
     - `data`: collection of data
     - `xlabel`: label for the x-axis
     - `title`: title for the plot
     - `xlim`: min and max x-axis values

    :postcondition: figures/<label>_cdf.png is a plot of data

    :return: PlotSetup
    """
    #if xlim is None:
    #    xlim = (min(0, min(data)), max(data))
    with PlotSetup(label=plot_label, title=title, legend_location=legend_location, plot_type='cdf',
                   axe=axe, figure=figure,
                   xlabel=xlabel, xlim=xlim, ylabel=ylabel) as plotter:
        data = numpy.array(data)
        data.sort()
        x = numpy.array(range(len(data)))/float(len(data))
        plotter.axe.plot(x, data,  *args, **kwargs)
    return plotter


def plot_boxplot(data, label='', plot_label='', axe=None, figure=None, xlabel='', title='', xlim=None,
                 legend_location='upper right',
                 boxcolor='b', *args, **kwargs):
    """
    Create a boxplot of the data

    :param:

     - `data`: data collection to plot
     - `label`: prefix for filename
     - `xlabel`: label for x-axis
     - `title`: title for plot
     - `plot_label`: label to add to plot for the legend

    :postcondition: figures/<label>_boxplot.png is a plot of data
    :return: path to plot
    """
    if xlim is None:
        xlim = (min(0, min(data)), max(data))
    with PlotSetup(label=plot_label, axe=axe, figure=figure, title=title, legend_location=legend_location,
                   xlabel=xlabel, xlim=xlim, plot_type='boxplot') as plotter:
        # boxplot wasn't implemented with the `label` keyword the way the other plots were
        bp = plotter.axe.boxplot(data, vert=False, *args, **kwargs)
        plt.setp(bp['boxes'], color=boxcolor, alpha=0.75)
        plt.setp(bp['medians'], color='r', alpha=0.75)
        plt.setp(bp['fliers'], markerfacecolor='none', markeredgecolor='r', marker='o', alpha=0.75)
    return plotter


# this will run forever, don't use it if large trials
def plot_kde(data, plot_label, label, axe=None, figure=None, xlabel=None, title=None, xlim=None,
             legend_location='upper right',
             *args, **kwargs):
    """
    Plots a kernel density estimate
    """
    if xlim is None:
        xlim = (min(0, min(data)), max(data))
    with PlotSetup(label=plot_label, title=title, xlabel=xlabel,
                   axe=axe, figure=figure, legend_location=legend_location,
                   xlim=xlim, plot_type='kde') as plotter:
        kde = gaussian_kde(data)
        x = numpy.linspace(min(data), max(data), len(data))
        plotter.axe.plot(x, kde(x), *args, **kwargs)
    return plotter


indent = "   {0},{1:.2f}"
def summary(data, title):    
    print ".. csv-table:: " + title
    print "   :header: Statistic, Value\n"
    print indent.format("Count", len(data))
    print indent.format("Min", min(data))
    q_1 = numpy.percentile(data, 25)
    print indent.format('Q1', q_1)
    print indent.format("Median", numpy.median(data))
    q_3 = numpy.percentile(data, 75)
    print indent.format("Q3", q_3)
    print indent.format("Max", max(data))
    print indent.format("IQR", q_3 - q_1)
    print indent.format("Mean", numpy.mean(data))
    print indent.format("STD", numpy.std(data))


indent = "   {0},{1}"
digits = "{0:.2f}"

def get_row(function, data):
    return ','.join((digits.format(function(data[column])) for column in data.columns))

def percentile(data, percentile):
    return ",".join((digits.format(numpy.percentile(data[column], percentile)) for column in data.columns))

def iqr(data):
    return ",".join((digits.format(numpy.percentile(data[column], 75)-numpy.percentile(data[column], 25)) for column in  data.columns))

def multisummary(data, title):
    """
    Create a side-by-side (csv) table

    :param:

     - `data`: collection of data collection
     - `title`: title to give the csv-table
    """
    header = 'Statistic,' +  ','.join([column for column in data.columns])
    print ".. csv-table:: " + title
    print "   :header: {0}\n".format(header)

    print indent.format("Count", get_row(len, data))
    print indent.format("Min", get_row(min, data))
    print indent.format('Q1', percentile(data, 25))
    print indent.format("Median", get_row(numpy.median, data))
    print indent.format("Q3", percentile(data, 25))
    print indent.format("Max", get_row(max, data))
    print indent.format("IQR", iqr(data))
    print indent.format("Mean", get_row(numpy.mean, data))
    print indent.format("STD", get_row(numpy.std, data))
    return


def outcome(thing_observed, amount_observed, probability, title, data, plot_label):
    """
    Prints a summary and plots the data

    :param:

     - `thing_observed`: A descriptive name of what you are calculating the probability for
     - `amount_observed`: What was the original value seen in the experiment?
     - `probability`: What was the simulated probability of the thing observed?
     - `title`: A descriptive title for the output
     - `data`: The simulated trial data
     - `plot_label`: A label to give the plot-filenames
    """
    print "**Observed {0}:** {1}\n".format(thing_observed, amount_observed)
    print "**Probability of {0} by chance:** {1:.2f}".format(thing_observed, probability)
    print

    summary(data, title)

    label = "Original {0}".format(thing_observed)
    threshold = amount_observed

    hline = plot_hline(y=probability,
                       plot_label=plot_label,
                       color='FireBrick',
                       label='Probability')

    vline = plot_vline(x=threshold,
                       plot_label=plot_label,
                       axe=hline.axe,
                       figure=hline.figure,
                       label=label)

    histogram = plot_histogram(data=data,
                           axe=vline.axe,
                           figure=vline.figure,
                           plot_label=plot_label,
                           xlabel='Count',
                           normed=True,
                           legend_location='upper right', 
                           histtype='stepfilled',
                           alpha=0.5,
                           bins=20,
                           title=title)

    print ".. figure:: " + histogram.path
    vline = plot_vline(x=1-probability, plot_label=plot_label,
                       label='Probability of {0}'.format(thing_observed),
                       color="g")

    hline = plot_hline(y=threshold, axe=vline.axe, figure=vline.figure,
                        plot_label=plot_label, label=label,
                        color='FireBrick', alpha=0.5)

    cdf = plot_cdf(data=data,
                   axe=hline.axe,
                   figure=hline.figure,
                   plot_label=plot_label,
                   legend_location='upper left',
                   xlabel='Fraction',
                   title=title)

    print ".. figure:: " + cdf.path

    vline = plot_vline(x=threshold,
                       plot_label=plot_label,
                       label=label,
                       color='FireBrick')
    boxplot = plot_boxplot(data=data,
                           legend_location='upper left',
                           axe=vline.axe,
                           figure=vline.figure,
                           plot_label=plot_label,
                           title=title)
    print ".. figure:: " + boxplot.path
    return



def elapsed_time(function):
    def _function(*args, **kwargs):
        start = time.time()
        outcome = function(*args, **kwargs)
        print "Elapsed Time: {0} Seconds".format(time.time() - start)
        return outcome
    return _function
