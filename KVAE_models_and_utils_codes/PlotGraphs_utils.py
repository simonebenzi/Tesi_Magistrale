
import matplotlib.pyplot as plt
from matplotlib import figure
#import gc
import numpy as np

colors_array = np.array(['lightpink', 'grey', 'blue', 'cyan', 'lime', 'green', 'yellow', 'gold', 'red', 'maroon',
                   'rosybrown', 'salmon', 'sienna', 'palegreen', 'sandybrown', 'deepskyblue', 
                   'fuchsia', 'purple', 'crimson', 'cornflowerblue', 
                   'midnightblue', 'mediumturquoise', 'bisque', 'gainsboro', 'indigo',
                   'white', 'coral', 'powderblue', 'cadetblue', 'orchid', 'burlywood', 'olive', 'lavender', 
                   'olivedrab', 'seashell', 'mistyrose', 'firebrick', 'dimgrey', 'tan', 'darkorange',
                   'tomato', 'dodgerblue', 'slateblue', 'rebeccapurple', 'moccasin'])

# states: these could be both the states or the derivatives:
# n_states_to_display: how many of the states given should be displayed
# file : where to save them
def scatter_states(states, n_states_to_display, file):
    
    x_axis = np.arange(len(states))
    fig, axes = plt.subplots(n_states_to_display)
    for i in range(0,n_states_to_display):
        axes[i].scatter(x_axis, states[:, i], s=1) 
    plt.savefig(file)
    #plt.show() 
    
    return

def plot_states(states, n_states_to_display, file):
    
    x_axis = np.arange(len(states))
    fig, axes = plt.subplots(n_states_to_display)
    for i in range(0,n_states_to_display):
        axes[i].plot(x_axis, states[:, i]) 
    plt.savefig(file)
    #plt.show() 
    return

# Functions for plotting the evolution of the losses
def plot_loss(loss, file, title = None):
    
    plt.close('all') # first close all other plots, or they will appear one over the others    
    fig = figure.Figure()
    ax = fig.subplots(1)
    ax.plot(loss) 
    
    if title != None:
        ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    
    fig.savefig(file)
    #plt.show() 
    #plt.close('all')
    
    return

def plot_predicted_vs_real_states(predicted_states, real_states, file):
    
    fig = figure.Figure()
    ax = fig.subplots(1)
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    ax.plot(real_states[:, 0], real_states[:, 1], 'red')
    ax.plot(predicted_states[:, 0], predicted_states[:, 1], 'black')
    fig.savefig(file)
    
    '''
    plt.cla() 
    plt.clf()
    plt.close(fig)
    plt.close('all')
    
    del fig, ax
    del predicted_states, real_states, file
    
    gc.collect()
    '''
    
    return

def plot_predicted_vs_real_states_on1D(predicted_states, real_states, n_states_to_display, file):
    
    x_axis = np.arange(len(real_states))
    fig, axes = plt.subplots(n_states_to_display)
    for i in range(0,n_states_to_display):
        axes[i].plot(x_axis, real_states[:, i], 'red') 
        axes[i].plot(x_axis, predicted_states[:, i], 'black') 
    plt.savefig(file)
    #plt.show() 
    
    return

def plot_predicted_vs_real_states_onScatterPlotWithQuivers(predicted_states, real_states, file):
    
    fig = figure.Figure()
    ax = fig.subplots(1)
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)

    ax.scatter(predicted_states[:, 0], predicted_states[:, 1], c = 'red', cmap="RdYlGn", s=45)
    ax.scatter(real_states[:, 0],      real_states[:, 1], c = 'blue', cmap="RdYlGn", s=45)
    ax.plot([real_states[:, 0], predicted_states[:, 0]],
            [real_states[:, 1], predicted_states[:, 1]])
    
    fig.savefig(file)

    return

def plot_clusters(cluster_assignment, odometry, file):
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    
    c = colors_array[cluster_assignment[:]]
    plt.scatter(odometry[:, 0], odometry[:, 1], c = c, cmap="RdYlGn", s=45)
    plt.savefig(file)
    return
    
def PlottingZVelocitiesAndTheirMeans(meanVelPerCluster, meanVels, clusterAssignments, file):
    
    fig = figure.Figure()
    ax = fig.subplots(1)
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    
    colorsPerCluster   = colors_array[0:meanVelPerCluster.shape[0]]
    colorsPerDataPoint = colors_array[clusterAssignments[:]]
    
    # z vel centers
    ax.scatter(meanVelPerCluster[:, 0], meanVelPerCluster[:, 1], c = colorsPerCluster, marker='*', s = 20)
    # z velocity means
    ax.scatter(meanVels[:, 0], meanVels[:, 1], c = colorsPerDataPoint, s = 5)   
    fig.savefig(file)      
    return

def plot_alpha_values(alpha_values_to_plot,file_name):    
    
    fig = plt.figure(figsize=[6, 6])
    ax = fig.gca()

    for i in range(alpha_values_to_plot.shape[1]):
        ax.plot(alpha_values_to_plot[:,i], linestyle='-')
    plt.title('alpha values')
    plt.savefig(file_name)
    #plt.show() 

def HandleLossOverAllEpochs(averageLossesOverAllEpochs, lossesOverCurrentEpoch, folderFileName):
    
    # Average the loss for current epoch
    lossesOverCurrentEpochAverage   = np.mean(lossesOverCurrentEpoch, axis=0)
    
    # Insert average of current epoch in array containing the averages of all epochs
    averageLossesOverAllEpochs.append(lossesOverCurrentEpochAverage)
    
    # From torch to numpy
    #averageLossesOverAllEpochsArray = np.asarray(averageLossesOverAllEpochs)
    
    # Plot the loss over all epochs
    plot_loss(averageLossesOverAllEpochs, folderFileName)
    

    return averageLossesOverAllEpochs