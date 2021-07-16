import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow as gpf
import pandas as pd
import os
import datetime

from BNQD import BNQD
from matplotlib import cm

plt.rc('axes', titlesize=24)        # fontsize of the axes title
plt.rc('axes', labelsize=18)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)       # fontsize of the tick labels
plt.rc('ytick', labelsize=12)       # fontsize of the tick labels
plt.rc('legend', fontsize=20)       # legend fontsize
plt.rc('figure', titlesize=32)      # fontsize of the figure title

print('TensorFlow version  ', tf.__version__)
print('NumPy version       ', np.__version__)
print('GPflow version      ', gpf.__version__)
print('BNQD version        ', BNQD.__version__)

datafolder = r'D:\SURFdrive\Projects\BNQD_MOGP\examples\Major depressive disorder'
datafile = r'ESMdata2.csv'

df = pd.read_csv(datafolder + os.path.sep + datafile, delimiter=',')

df = df.rename(columns={'Unnamed: 0': 'id'})

N = len(df.index)
seconds_in_day = 60*60*24

dates_to_datetime = [datetime.datetime.strptime(date, '%d/%m/%y') for date in df['date'].values]
measure_times = [datetime.datetime.strptime(time, '%H:%M:%S') for time in df['resptime_s'].values]

precise_times = [datetime.datetime(year=dates_to_datetime[i].year,
                                   month=dates_to_datetime[i].month,
                                   day=dates_to_datetime[i].day,
                                   hour=measure_times[i].hour,
                                   minute=measure_times[i].minute)  for i in range(N)]

timedelta_from_day0 = [time - precise_times[0] for time in precise_times]
days_from_day0 = np.array([td.days + td.seconds / seconds_in_day for td in timedelta_from_day0])
x = days_from_day0

ad_dosage = df['concentrat']
phases = df['phase']

colors_phases = cm.get_cmap('tab10', 10)
phase_labels = ['Baseline', 'Double blind before reduction', 'Double blind during reduction', 'Post medication reduction', 'Post experiment']

mood_items = ['mood_relaxed', 'mood_down', 'mood_irritat', 'mood_satisfi',
             'mood_lonely', 'mood_anxious', 'mood_enthus', 'mood_suspic',
             'mood_cheerf', 'mood_guilty', 'mood_doubt', 'mood_strong']

mood_needs_shift = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0])

colors_mood = cm.get_cmap('rainbow', len(mood_items))

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 16))
ax = axes[0]
for i, phase in enumerate(np.arange(1, 6)):
    ix_start = df[df['phase'] == phase].first_valid_index()
    ix_end = df[df['phase'] == phase+1].first_valid_index()
    if ix_end is None:
        ix_end = N-1
    ax.axvspan(xmin=x[ix_start], xmax=x[ix_end],
               alpha=0.2, color=colors_phases(i), label=phase_labels[i])
ax.plot(x, ad_dosage, color='k', lw=2, label='AD dosage')
ax.set_ylabel('AD dosage')
ax.set_xlim([0, x[-1]])
ax.legend()
ax.set_title('Anti-depressant dosage across experimental phases')

ax = axes[1]

for i, mood in enumerate(mood_items):
    y = df[mood] + 3*mood_needs_shift[i]  # same Likert everywhere
    ax.plot(x, y, ls=':', color=colors_mood(i), label=mood)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Mood score')
ax.set_xlim([0, x[-1]])
ax.legend(fontsize=14, ncol=3)
ax.set_title('Self-reported mood scores')
plt.show()

### subset range
#
# select_phases = [1, 2, 3]
# ix_min = N
# ix_max = 0
#
# fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 16))
# ax = axes[0]
# for phase in select_phases:
#     ix_start = df[df['phase'] == phase].first_valid_index()
#     ix_end = df[df['phase'] == phase+1].first_valid_index()
#     if ix_end is None:
#         ix_end = N-1
#     if ix_start < ix_min:
#         ix_min = ix_start
#     if ix_end > ix_max:
#         ix_max = ix_end
#     ax.axvspan(xmin=x[ix_start], xmax=x[ix_end],
#                alpha=0.2, color=colors_phases(phase-1), label=phase_labels[phase-1])
# ax.plot(x[ix_min:ix_max], ad_dosage[ix_min:ix_max], color='k', lw=2, label='AD dosage')
# # ax.set_xlabel('Time (days)')
# ax.set_ylabel('AD dosage')
# ax.set_xlim([x[ix_min], x[ix_max]])
# ax.legend()
#
# ax = axes[1]
#
# for i, mood in enumerate(mood_items):
#     y = df[mood] + 3*mood_needs_shift[i]  # same Likert everywhere
#     ax.plot(x[ix_min:ix_max], y[ix_min:ix_max], ls=':', color=colors_mood(i), label=mood)
# ax.set_xlabel('Time (days)')
# ax.set_ylabel('Mood score')
# # ax.set_xlim([0, x[-1]])
# ax.legend(fontsize=14, ncol=3)
# plt.show()
