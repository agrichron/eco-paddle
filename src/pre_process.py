from pathlib import Path
import os
import cv2
import pickle
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

data_path = Path('/home/aistudio/data')
dataset_path_root = data_path / 'UCF-101'
dataset_path = '/home/aistudio/data/data48916/UCF-101.zip'

def unrar(filename):

    # 解压数据
    if not data_path.exists():
        data_path.mkdir()
    
    if not dataset_path_root.exists():
        dataset_path_root.mkdir()
    
    # os.system(f'rar x {filename} {data_path}')
    os.system(f'unzip {filename} -d {data_path}')

def video2jpg(sample):

    print(f'convert {sample}.')

    jpg_dir = sample.with_suffix('')
    jpg_dir.mkdir(parents=True)

    cap = cv2.VideoCapture(str(sample))
    frame_count = 1
    ret = True
    while ret:
        ret, frame = cap.read()
        
        if ret:
            cv2.imwrite(str(jpg_dir / (sample.stem + '_%04d.jpg' % frame_count )), frame)
            frame_count += 1
    
    cap.release()
    sample.unlink()

def generate_train_val_list():
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'

    if not train_dir.exists():
        train_dir.mkdir()
    if not val_dir.exists():
        val_dir.mkdir()

    classInd_file = Path('config/classInd.txt')
    trian_list = Path('config/trainlist01.txt')
    test_list = Path('config/testlist01.txt')

    classInd = {}
    with open(classInd_file) as ff:
        for line in ff:
            line = line.strip()
            if not line:
                continue
                
            index, name = line.split()
            index = int(index) - 1
            classInd[name] = index

    with open(trian_list) as ff:
        for line in ff:
            line = line.strip()
            if not line:
                continue
            avi, index = line.split()
            index = int(index) - 1
            avi = dataset_path_root / Path(avi).with_suffix('')

            frame = [str(x) for x in sorted(avi.glob('*jpg'))]


            pkl_name = avi.with_suffix('.pkl').name
            out_pkl = train_dir / pkl_name
            f = open(out_pkl, 'wb')
            pickle.dump((avi.stem, index, frame), f, -1)
            f.close()
    
    with open(test_list) as ff:
        for line in ff:
            line = line.strip()
            if not line:
                continue
            avi = line
            index = classInd[Path(avi).parent.stem]
            avi = dataset_path_root / Path(avi).with_suffix('')

            frame = [str(x) for x in sorted(avi.glob('*jpg'))]


            pkl_name = avi.with_suffix('.pkl').name
            out_pkl = val_dir / pkl_name
            f = open(out_pkl, 'wb')
            pickle.dump((avi.stem, index, frame), f, -1)
            f.close()


    train_content = '\n'.join(map(str, train_dir.glob('*.pkl')))
    val_content = '\n'.join(map(str, val_dir.glob('*.pkl')))
    (data_path / 'train.list').write_text(train_content)
    (data_path / 'val.list').write_text(val_content)


def func(args):
    '''
    cls: class name
    sample: video name
    '''
    cls, sample = args

    avi_path_jpg = dataset_path_root / (cls + '_jpg') / sample.stem
    print(f'doing: convert {sample} to {avi_path_jpg}')
    
    cap = cv2.VideoCapture(str(sample))
    frame_count = 1
    ret = True
    while ret:
        ret, frame = cap.read()
        
        if ret:
            cv2.imwrite(str(avi_path_jpg / (sample.stem + '_%d.jpg' % frame_count )), frame)
        frame_count += 1
    
    cap.release()
    sample.unlink()

def jpg2pkl_and_gen_data_list():
    train_dir = dataset_path_root / 'train'
    val_dir = dataset_path_root / 'val'

    if not train_dir.exists():
        train_dir.mkdir()
    if not val_dir.exists():
        val_dir.mkdir()


    classInd_file = Path('config/classInd.txt')
    trian_list = Path('config/trainlist01.txt')
    test_list = Path('config/testlist01.txt')

    classInd = {}
    with open(classInd_file) as ff:
        for line in ff:
            line = line.strip()
            if not line:
                continue
                
            index, name = line.split()
            index = int(index) - 1
            classInd[name] = index

    with open(trian_list) as ff:
        for line in ff:
            line = line.strip()
            if not line:
                continue
            avi, index = line.split()
            index = int(index) - 1
            avi = dataset_path_root / (Path(avi).parent.stem + '_jpg') / Path(avi).stem

            frame = [str(x) for x in sorted(avi.glob('*jpg'))]


            pkl_name = avi.with_suffix('.pkl').name
            out_pkl = train_dir / pkl_name
            f = open(out_pkl, 'wb')
            pickle.dump((avi.stem, index, frame), f, -1)
            f.close()
    
    with open(test_list) as ff:
        for line in ff:
            line = line.strip()
            if not line:
                continue
            avi = line
            index = classInd[Path(avi).parent.stem]
            avi = dataset_path_root / (Path(avi).parent.stem + '_jpg') / Path(avi).stem

            frame = [str(x) for x in sorted(avi.glob('*jpg'))]


            pkl_name = avi.with_suffix('.pkl').name
            out_pkl = val_dir / pkl_name
            f = open(out_pkl, 'wb')
            pickle.dump((avi.stem, index, frame), f, -1)
            f.close()


    
    train_content = '\n'.join(map(str, train_dir.glob('*.pkl')))
    val_content = '\n'.join(map(str, val_dir.glob('*.pkl')))
    (dataset_path_root / 'train.list').write_text(train_content)
    (dataset_path_root / 'val.list').write_text(val_content)



if __name__ == '__main__':

    # unrar dataset

    # unrar(dataset_path)
    
    videos = []
    for cls in dataset_path_root.iterdir():
        for avi in cls.glob('*avi'):
            videos += [avi]
    
    # avi2jpg
    pool = Pool(processes=cpu_count())
    pool.map_async(video2jpg, videos)
    pool.close()
    pool.join()
    print('avi2jpg 结束。')

    generate_train_val_list()
 
