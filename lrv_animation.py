import pandas as pd

from matplotlib.colors import Normalize
import configparser as ConfigParser
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import gridspec
from matplotlib.patches import Wedge

from matplotlib.patches import Polygon

par = {'radius_dish':42.5,
            'fig_width':100,
            'animation_dpi':50,
'font_size':12,
'movie_dir':'.',
'dt':1/16
        }
font_size=12

def rotate_vector_clockwise(angle, vector):
    '''postive angle rotates clockwise,
    negative angle rotates counter-clockwise'''
    return [
        np.cos(angle) * vector[0] + np.sin(angle) * vector[1],
        -np.sin(angle) * vector[0] + np.cos(angle) * vector[1]
    ]


def clockwise_angle_from_first_to_second_vector(first_vector, second_vector):
    '''clockwise change from first to second counts as positive,
    counter-clockwise change counts as negative'''

    angle_tmp = (np.arctan2(second_vector[1], second_vector[0]) -
                 np.arctan2(first_vector[1], first_vector[0]))
    angle_tmp = (angle_tmp - (np.abs(angle_tmp) > np.pi) *
                 np.sign(angle_tmp) * 2. * np.pi)

    return -angle_tmp

def animate_track(track, par, speed=1.0, zoom_dx=3, save_movie=False,
                      movie_name='some_movie_name'):
        '''animates track'''


        old_dpi = plt.rcParams['savefig.dpi']
        plt.rcParams['savefig.dpi'] = par['animation_dpi']
        # figure settings
        fig = plt.figure(figsize=(par['fig_width'], par['fig_width']))
        gs1 = gridspec.GridSpec(1, 1)
        gs1.get_subplot_params(fig)
        gs1.update(left=0.02, right=0.98,
                   hspace=0.1, wspace=0.1, bottom=0.02, top=0.98)
        ax1 = plt.subplot(gs1[0, 0])
        [ax1.spines[str_tmp].set_color('none')
         for str_tmp in ['top', 'right', 'left', 'bottom']]
        plt.setp(ax1, xlim=(-par['radius_dish'] - 5, par['radius_dish'] + 5),
                 ylim=(-par['radius_dish'] - 5,
                       par['radius_dish'] + 5), xticks=[], yticks=[])

        # plot edge of petri dish
        #patch = Wedge((.0, .0), par['radius_dish'] + 3, 0, 360, width=3,
        #              fc='k', lw=0, alpha=0.1)
        #ax1.add_artist(patch)



        # time text
        time_text = ax1.text(x=0.03, y=0.03, s='',
                             size=font_size, horizontalalignment='left',
                             verticalalignment='bottom',
                             alpha=1, transform=ax1.transAxes)

        # valid text
        valid_text = ax1.text(x=0.03, y=0.93, s='', size=font_size,
                              horizontalalignment='left',
                              verticalalignment='bottom',
                              alpha=1, color='r', transform=ax1.transAxes)

        # back_vector_orthogonal
        back_vector_orthogonal = np.array(rotate_vector_clockwise(
            np.pi / 2.,
            [track.tail_vector_x,track.tail_vector_y])).T
        print(back_vector_orthogonal.shape)

        # init subplot
        midpoint_line, = ax1.plot([], [], 'k-', alpha=0.5, lw=2)
        head_line, = ax1.plot([], [], 'm-', alpha=0.5, lw=2)

        head_line_left, = ax1.plot([], [], 'g-', alpha=0.5, lw=10)
        head_line_right, = ax1.plot([], [], 'r-', alpha=0.5, lw=10)

        current_spine, = ax1.plot([], [], 'k-', lw=1)

        contour_line = Polygon(np.nan * np.zeros((2, 2)), lw=1,
                               fc='lightgray', ec='k')
        ax1.add_artist(contour_line)


        # initialization function: plot the background of each frame
        def init():
            midpoint_line.set_data([], [])
            head_line.set_data([], [])
            head_line_left.set_data([], [])
            head_line_right.set_data([], [])
            current_spine.set_data([], [])
            contour_line.set_xy(np.nan * np.zeros((2, 2)))
            return (midpoint_line, head_line, head_line_left, head_line_right,
                    current_spine, contour_line,)

        # animation function
        def animate(i):

            #midpoint_line.set_data(self.spine[4][:i, 0], self.spine[4][:i, 1])
            #head_line.set_data(self.spine[11][:i, 0], self.spine[11][:i, 1])

            current_spine.set_data(
                [track['spinepoint_x_'+str(idx+1)][i] for idx in range(12)],
                [track['spinepoint_y_'+str(idx+1)][i] for idx in range(12)]
            )

            contour_data = np.array([[track['contourpoint_x_'+str(idx+1)][i],
                                          track['contourpoint_y_'+str(idx+1)][i]]
                                          for idx in range(22)])
            #print(contour_data)
            contour_line.set_xy(contour_data)
            cmap = cm.autumn
            norm = Normalize(vmin=0, vmax=2)
            
            contour_line.set_color(cmap(norm(track.tail_speed_forward[i])))
            # time text
            time_text.set_text(str(np.round(track.frame[i], 1)) + ' seconds')

            # valid text
            #if np.isnan(self.valid_frame[i]):
            #    valid_text.set_text('Invalid frame')
            #else:
            #    valid_text.set_text('')

            # zoom in
            if True:
                plt.setp(ax1,
                         xlim=(track['spinepoint_x_5'][i] - zoom_dx,
                               track['spinepoint_x_5'][i]+ zoom_dx),
                         ylim=(track['spinepoint_y_5'][i] - zoom_dx,
                               track['spinepoint_y_5'][i] + zoom_dx))

            # if step
            if(False):
                if i in self.step_idx:
                    ax1.plot(
                        [self.spine[0][i, 0] - 0.5 * back_vector_orthogonal[i, 0],
                         self.spine[0][i, 0] + 0.5 * back_vector_orthogonal[i, 0]],
                        [self.spine[0][i, 1] - 0.5 * back_vector_orthogonal[i, 1],
                         self.spine[0][i, 1] + 0.5 * back_vector_orthogonal[i, 1]],
                        'k-', alpha=0.2, lw=4)

            return (midpoint_line, head_line, current_spine, contour_line,)

        global ani
        ani = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(track.frame),
            interval=int(1000. / speed) * float(par['dt']),
            repeat=False)

        # save or show movie
        if True:
            print('Saving movie...')

            # bitrate = 100 - 600 works fine
            mywriter = animation.FFMpegWriter(fps = np.round(1 / float(par['dt'])))

            ani.save(par['movie_dir'] +
                     '/' + movie_name + '.mp4',
                     writer=mywriter)

            print('...done')

        plt.rcParams['savefig.dpi'] = old_dpi

data = pd.read_csv("with_contours_analyzed.csv")
track = data[data.id==1][0:100]
print(track.columns)
animate_track(track,par)
