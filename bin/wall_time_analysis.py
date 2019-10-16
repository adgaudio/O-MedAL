import pandas as pd
import dateutil.parser
import datetime as dt
import time

# get wall time at last epoch of the AL iter for all models.

# the 0.125 model, 80.08% labeled corresponds to al_iter=38
df = pd.read_csv('data/_analysis/RMO6-12.5e-20191004T232324.log/logdata.csv')
df['t2'] = df['time'] - time.mktime(dateutil.parser.parse('Fri Oct  4 23:23:24 EDT 2019').timetuple())
print('OMedAL p=0.125, 80.08% labeled. wall time:',  dt.timedelta(seconds=df.query('al_iter==38')['t2'].max()))

# OMedAL 0.875 model
df = pd.read_csv('data/_analysis/RMO6-87.5e-20191004T232324.log/logdata.csv')
df['t2'] = df['time'] - time.mktime(dateutil.parser.parse('Fri Oct  4 23:23:24 EDT 2019').timetuple())
# at 80.08% labeled
print('OMedAL p=0.875, 80.08% labeled. wall time:',  dt.timedelta(seconds=df.query('al_iter==38')['t2'].max()))
# at 40.04% labeled  (may be replaced by a dedicated labeling efficient model)
print('OMedAL p=0.875, 40.04% labeled. wall time:',  dt.timedelta(seconds=df.query('al_iter==19')['t2'].max()))

# OMedAL patience=5
df = pd.read_csv('data/_analysis/RMO6-87.5-5patience-20191005T215029.log/logdata.csv')
df['t2'] = df['time'] - time.mktime(dateutil.parser.parse('Sat Oct  5 21:50:29 EDT 2019').timetuple())
print('OMedAL p=0.875, patience=05, 90.62% labeled. wall time:',
      dt.timedelta(seconds=df.query('al_iter==43')['t2'].max()))

# OMedAL patience=10
df = pd.read_csv('data/_analysis/RMO6-87.5-10patience-20191006T004721.log/logdata.csv')
df['t2'] = df['time'] - time.mktime(dateutil.parser.parse('Sun Oct  6 00:47:21 EDT 2019').timetuple())
print('OMedAL p=0.875, patience=10, 67.44% labeled. wall time:',
      dt.timedelta(seconds=df.query('al_iter==32')['t2'].max()))

# OMedAL patience=20
df = pd.read_csv('data/_analysis/RMO6-87.5-20patience-20191006T133459.log/logdata.csv')
df['t2'] = df['time'] - time.mktime(dateutil.parser.parse('Sun Oct  6 13:34:59 EDT 2019').timetuple())
print('OMedAL p=0.875, patience=20, 25.29% labeled. wall time:',
      dt.timedelta(seconds=df.query('al_iter==12')['t2'].max()))

# resnet18 baseline
df = pd.read_csv('data/_analysis/R6b-20191005T013517.log/logdata.csv')
df['t2'] = df['time'] - time.mktime(dateutil.parser.parse('Sat Oct  5 01:35:17 EDT 2019').timetuple())
print('Resnet18 wall time:',  dt.timedelta(seconds=df['t2'].max()))

# MedAL patience 10
df = pd.read_csv('data/_analysis/RM6i-20191005T015713.log/logdata.csv')
df['t2'] = df['time'] - time.mktime(dateutil.parser.parse('Sat Oct  5 01:31:33 EDT 2019').timetuple())
print('MedAL patience=10. wall time:',  dt.timedelta(seconds=df.query('al_iter==43')['t2'].max()))

# MedAL patience 20
df = pd.read_csv('data/_analysis/RM6h-20191005T040008.log/logdata.csv')
df['t2'] = df['time'] - time.mktime(dateutil.parser.parse('Sat Oct  5 04:00:08 EDT 2019').timetuple())
# --> most accurate
print('MedAL patience=20. wall time (max accuracy):',  dt.timedelta(seconds=df.query('al_iter==35')['t2'].max()))
# --> most labeling efficient
print('MedAL patience=20. wall time (most labeling efficient):',  dt.timedelta(seconds=df.query('al_iter==16')['t2'].max()))
