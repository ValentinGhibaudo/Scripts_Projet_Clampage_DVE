import sys,os
import getpass
from pathlib import Path

if getpass.getuser() in ('valentin.ghibaudo','baptiste.balanca') and  sys.platform.startswith('linux'):
    base_folder_neuro_rea = '/crnldata/tiger/baptiste.balanca/Neuro_rea_monitorage/'

base_folder_neuro_rea = Path(base_folder_neuro_rea)
base_folder = base_folder_neuro_rea / 'these_ayoub_clampage_dve'
data_path = base_folder / 'data'

base_mnt_data = Path('/mnt/data/valentinghibaudo/')
precomputedir = base_mnt_data / 'precompute_these_ayoub' # mnt/data/

if __name__ == '__main__':
    print(base_folder, base_folder.exists())
    print(data_path, data_path.exists())
    print(precomputedir, precomputedir.exists())