import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

data_yes_cam = [3,1,2,0,2,0,0,5,0,0,1,2,1,7,2,1,0,2,1,1,0,0,1,1,0,3,3,0,1,0,0,2,3,0,0,1,1,2,1,3,0,1,0,3,3,0,0,0,1,0,0,0,0,0,3,0,1,2]
data_yes_pv = [10,1,4,6,7,2,3,0,7,6,3,5,7,4,5,3,4,4,8,8,5,10,8,5,0,4,6,0,5,3,9,4,6,2,7,5,6,8,0,6]

data_no_cam = [15,15,16,15,15,14,16,15,16,16,16,16,15,13,16,16,15,16,16,16,14,15,16,16,16,16,15,16,15,15,15,16,13,16,16,16,14,15,14,16,16,16,15,16,15,15,14,16,15,16,14,15,16,14,16,16,16,16]
data_no_pv = [10,15,12,11,11,8,13,8,12,10,16,10,14,16,9,11,13,14,8,11,10,10,10,12,15,14,13,16,12,13,10,16,13,14,13,10,12,16,15,7]

bins_yes = np.linspace(0, 14, 14)
bins_no = np.linspace(0, 16, 16)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 24}

matplotlib.rc('font', **font)


dfig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True,figsize=(24,6))
axs[0].set_ylabel('Participants (count)')
axs[1].set_ylabel('Participants (count)')

# We can set the number of bins with the `bins` kwarg
axs[0].hist([data_yes_cam, data_yes_pv], bins = bins_no)
axs[1].hist([data_no_cam, data_no_pv], bins = bins_no)
axs[0].title.set_text('Model\'s prediction is incorrect')
axs[1].title.set_text('Model\'s prediction is correct')
axs[0].set_xlabel("Participant's correct answers (out of 14)")
axs[1].set_xlabel("Participant's correct answers (out of 16)")
#axs[0].set_xticks(bins_yes)
#axs[1].set_xticks(bins_no)
dfig.subplots_adjust(bottom=0.3)
dfig.legend(labels=['Grad-CAM', 'PV'], title='Survey type', bbox_to_anchor=(0.5, 0.5), loc='upper left')
plt.show()