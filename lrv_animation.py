import pandas as pd
import matplotlib
from matplotlib.colors import Normalize
import configparser as ConfigParser
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import gridspec
from matplotlib.patches import Wedge
from matplotlib.patches import Polygon
import argparse
import os
import numpy as np, math, matplotlib.patches as patches
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


def animate_track(data_path,
                  id_,
                  attribute,
                  vmin,vmax, 
                  start = None,
                  end = None,
                  attribute_name = None,
                  save_as = None,
                  colormap = None,
                  zoom =True,
                  speed=1.0, 
                  zoom_dx=3
                  ):
    
    
        #plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg/bin'
        
        data = pd.read_csv(data_path)
        if(not(id_ in np.unique(data["id"]))):
            print("ID is not in the following ids!")
            print(np.unique(data["id"]))
            print("Choosing the id with index "+ str(id_) +" which is "+str(data.id[id_]))
            track = data[data.id ==data.id[id_]]
        else:           
            track = data[data.id ==id_]
            
        if(save_as==None):
             save_as = "larval_animation.mp4"
        if(start==None):
            start_frame = 0
        else:
            start_frame = int(round(start*16))
        if(end==None):
            track = track[start_frame:]
        else:
            end_frame = int(round(end*16))
            track = track[start_frame:end_frame]
            
        if(colormap==None):
            colormap = "bwr"
            
        par = {'radius_dish':42.5,
            'fig_width':100,
            'animation_dpi':200,
            'font_size':12,
            'movie_dir':os.getcwd(),
            'dt':1/16
            }
        #par["movie_dir"]=""
        font_size=12
        '''animates track'''
        if(attribute_name ==None):
            attribute_name = attribute

        old_dpi = plt.rcParams['savefig.dpi']
        plt.rcParams['savefig.dpi'] = par['animation_dpi']
        # figure settings
        #fig = plt.figure(figsize=(par['fig_width'], par['fig_width']))
        
        plt.rcParams["figure.figsize"] = (20,10)
        fig,[ax1,cax] = plt.subplots(1,2, gridspec_kw={"width_ratios":[150,10]})
        
        #gs1 = gridspec.GridSpec(1, 1)
        #gs1.get_subplot_params(fig)
        #gs1.update(left=0.02, right=0.98,
        #           hspace=0.1, wspace=0.1, bottom=0.02, top=0.98)
        #ax1 = plt.subplot(gs1[0, 0])
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
        #back_vector_orthogonal= np.array([track.tail_vector_x,track.tail_vector_y]).T
        back_vector_orthogonal = np.array(rotate_vector_clockwise(
            np.pi / 2.,
            [track.tail_vector_x,track.tail_vector_y])).T

        # init subplot
        midpoint_line, = ax1.plot([], [], 'k-', alpha=0.5, lw=2)

        head_line, = ax1.plot([], [], 'bo', alpha=0.5, lw=5)
        head_line_left, = ax1.plot([], [], 'g-', alpha=0.5, lw=10)
        head_line_right, = ax1.plot([], [], 'r-', alpha=0.5, lw=10)

        current_spine, = ax1.plot([], [], 'k-', lw=1)

        contour_line = Polygon(np.nan * np.zeros((2, 2)), lw=1,
                               fc='lightgray', ec='k')
        ax1.add_artist(contour_line)
        if(vmin<0):
            cmap = cm.get_cmap("bwr")
        else:
            cmap = cm.get_cmap("viridis")
            
        norm = Normalize(vmin=vmin, vmax=vmax)
        cb =matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')

        cb.ax.set_ylabel(attribute_name, rotation=270, labelpad=20)
        # initialization function: plot the background of each frame


        #head_line = plt.Arrow(head_data[0],head_data[1],head_data[2]-head_data[0],head_data[3]-head_data[1] )
        
        def init():
            #head_line = patches.Arrow(0,0,0,0 )
            #ax1.add_patch(head_line)
            midpoint_line.set_data([], [])
            head_line.set_data([],[])
            head_line_left.set_data([], [])
            head_line_right.set_data([], [])
            current_spine.set_data([], [])
            contour_line.set_xy(np.nan * np.zeros((2, 2)))
            return (midpoint_line,head_line, head_line_left, head_line_right,
                    current_spine, contour_line)
        
        spinepoints = [['spinepoint_x_'+str(idx+1)+"_conv"]+['spinepoint_y_'+str(idx+1)+"_conv"] for idx in range(12)]
        contourpoints = [['contourpoint_x_'+str(idx+1)+"_conv"]+['contourpoint_y_'+str(idx+1)+"_conv"] for idx in range(22)]
        columns = ["frame"]+spinepoints + contourpoints + [attribute]
        def flat2gen(alist):
          for item in alist:
            if isinstance(item, list):
              for subitem in item: yield subitem
            else:
              yield item
        columns = list(flat2gen(columns))
        
        track = track[columns].dropna()
        track.index = range(0,len(track.index))
        # animation function
        def animate(i):
            
            #midpoint_line.set_data(self.spine[4][:i, 0], self.spine[4][:i, 1])
            
           
            if(not(i in np.where(np.isnan(np.array(track.spinepoint_x_1_conv)))[0])):
                current_spine.set_data(
                    [track['spinepoint_x_'+str(idx+1)+"_conv"][i] for idx in range(12)],
                    [track['spinepoint_y_'+str(idx+1)+"_conv"][i] for idx in range(12)]
                )

                head_data = np.array([track['spinepoint_x_9_conv'].iloc[i], 
                                      track['spinepoint_y_9_conv'].iloc[i],
                                      track['spinepoint_x_11_conv'].iloc[i], 
                                      track['spinepoint_y_11_conv'].iloc[i]])
                #head_data = head_data.reshape((i,2))
                #head_data = head_data.T
                head_line.set_data([head_data[2]],[head_data[3]])
                #print(head_data)
                #global head_line
                #ax1.patches.remove(head_line)

               
                
                contour_data = np.array([[track['contourpoint_x_'+str(idx+1)+"_conv"].iloc[i],
                                              track['contourpoint_y_'+str(idx+1)+"_conv"].iloc[i]]
                                              for idx in range(22)])
               

                contour_line.set_xy(contour_data)
               
                contour_line.set_color(cmap(norm(track[attribute].iloc[i])))
                # time text
                time_text.set_text(str(np.round(track.frame[i]/16, 1)) + ' seconds')
                
                #ax1.patches.pop(0) 
                #head_line = plt.Arrow(head_data[0],head_data[1],head_data[2]-head_data[0],head_data[3]-head_data[1])
                #ax1.add_patch(head_line)
                # valid text
                #if np.isnan(self.valid_frame[i]):
                #    valid_text.set_text('Invalid frame')
                #else:
                #    valid_text.set_text('')
    
                # zoom in
                if zoom:
                    plt.setp(ax1,
                             xlim=(track['spinepoint_x_5_conv'].iloc[i] - zoom_dx,
                                   track['spinepoint_x_5_conv'].iloc[i]+ zoom_dx),
                             ylim=(track['spinepoint_y_5_conv'].iloc[i] - zoom_dx,
                                   track['spinepoint_y_5_conv'].iloc[i] + zoom_dx))
    
                # if step ( do not show steps at the moment)
                if(False):
                    if i in np.where(track.step_boolean==True)[0]:
                        ax1.plot(
                                [track.spinepoint_x_1_conv[i] - 0.5 * back_vector_orthogonal[i, 0],
                                 track.spinepoint_x_1_conv[i] + 0.5 * back_vector_orthogonal[i, 0]],
                                 [track.spinepoint_y_1_conv[i] - 0.5 * back_vector_orthogonal[i, 1],
                                  track.spinepoint_y_1_conv[i] + 0.5 * back_vector_orthogonal[i, 1]],
                                  'k-', alpha=0.2, lw=4,color = cmap(norm(track[attribute][i])))

            return (midpoint_line,  current_spine, contour_line,head_line)

        global ani
        ani = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(track.frame),
            interval=int(1000. / speed) * float(par['dt']),
            repeat=False)

        print('Saving movie... this may take some time')

        # bitrate = 100 - 600 works fine
        mywriter = animation.FFMpegWriter(fps = np.round(1 / float(par['dt'])))
        
        save_in = par['movie_dir'] +'/' + save_as + '.mp4'
        #print(save_in)
        #print(plt.rcParams['animation.ffmpeg_path'])
        save_in = save_in.replace("\\","/")
        #save_in = save_as
        ani.save(save_in,
                 writer=mywriter)

        print('...done')

        plt.rcParams['savefig.dpi'] = old_dpi




parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="Path to the analyzed data csv file.",
                    type=str)
parser.add_argument("id", help="larva id to be used",
                    type=int)
parser.add_argument("attr", help="name of the column for the color code",
                    type=str)
parser.add_argument("vmin", help="minimum value for the color scale limits",
                    type=float)
parser.add_argument("vmax", help="maximum value for the color scale limits",
                    type=float)
parser.add_argument("--start", help="start point of animation in seconds",
                    type=float)
parser.add_argument("--end", help="end point of animation in seconds",
                    type=float)
parser.add_argument("--attr_name", help="name of the attribute",
                    type=str)
parser.add_argument("--save_as", help="name of the mp4 file",
                    type=str)

args = parser.parse_args()
animate_track(args.data_path,args.id,args.attr,args.vmin,args.vmax,args.start,args.end, args.attr_name,args.save_as)
#python3 lrv_animation.py "data/with_contours_analyzed.csv" 1 "tail_speed_forward"  0 1.5 --start 0 --end 10 --attr_name "Tail velocity forward"

