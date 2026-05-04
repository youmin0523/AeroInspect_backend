"""새 데이터셋 → 20종 하자 매핑 통합 스크립트"""
import shutil, os, random, csv
from pathlib import Path
import numpy as np
from PIL import Image

random.seed(42)
RAW = Path('gdrive_raw')
DS = Path('datasets')

print('=' * 50)
print('  새 데이터셋 → 20종 하자 매핑 통합')
print('=' * 50)

# 1. Building walls (1,416장) → structural
print('\n--- 1. Building walls → structural ---')
CLASS_REMAP = {0: 0, 1: 1, 2: 1, 3: 0, 4: 1}
src = RAW / '정리_building_walls'
added = 0
for ss, sd in [('train', 'train')]:
    si = src / ss / 'images'
    sl = src / ss / 'labels'
    di = DS / 'structural' / 'images' / sd
    dl = DS / 'structural' / 'labels' / sd
    if not si.exists():
        continue
    for img in si.glob('*'):
        if img.suffix.lower() not in ('.jpg', '.png'):
            continue
        lbl = sl / (img.stem + '.txt')
        if not lbl.exists():
            continue
        shutil.copy2(img, di / f'bw_{img.name}')
        with open(lbl) as f:
            lines = f.readlines()
        with open(dl / f'bw_{img.stem}.txt', 'w') as f:
            for line in lines:
                p = line.strip().split()
                if len(p) >= 5:
                    c = CLASS_REMAP.get(int(p[0]), 0)
                    f.write(f'{c} {" ".join(p[1:])}\n')
        added += 1
print(f'  추가: {added}장')

# 2. Crack folder (9,856장) → normal/good + structural_crops
print('\n--- 2. Crack folder → normal + structural_crops ---')
cf = RAW / '정리_crack_folder'
neg, pos = 0, 0
for split in ['train', 'valid', 'test']:
    sd = 'val' if split == 'valid' else split
    nd = cf / split / 'Negative'
    if nd.exists():
        for img in nd.glob('*'):
            shutil.copy2(img, DS / 'normal' / 'good' / f'cf_{img.name}')
            neg += 1
    pd = cf / split / 'Positive'
    dst = DS / 'structural_crops' / sd / 'crack_indicator'
    dst.mkdir(parents=True, exist_ok=True)
    if pd.exists():
        imgs = list(pd.glob('*'))
        random.shuffle(imgs)
        for img in imgs[:500]:
            shutil.copy2(img, dst / f'cf_{img.name}')
            pos += 1
print(f'  normal/good: +{neg}장, crack_indicator: +{pos}장')

# 3. Dirtvision (2,121장) → surface
print('\n--- 3. Dirtvision → surface ---')
dv = RAW / '정리_dirtvision'
da = 0
for ss, sd in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
    si = dv / ss / 'images'
    sl = dv / ss / 'labels'
    di = DS / 'surface' / 'images' / sd
    dl = DS / 'surface' / 'labels' / sd
    if not si.exists():
        continue
    for img in si.glob('*.jpg'):
        lbl = sl / (img.stem + '.txt')
        if not lbl.exists():
            continue
        shutil.copy2(img, di / f'dirt_{img.name}')
        with open(lbl) as f:
            lines = f.readlines()
        with open(dl / f'dirt_{img.stem}.txt', 'w') as f:
            for line in lines:
                p = line.strip().split()
                if len(p) >= 5:
                    f.write(f'0 {" ".join(p[1:])}\n')
        da += 1
print(f'  추가: {da}장')

# 4. Crack900 (914장 RGB+IR+온도+마스크) → thermal/ U-Net
print('\n--- 4. Crack900 → thermal/ U-Net ---')
c9 = RAW / 'M4_M5_M3' / 'Crack900' / 'data'
td = DS / 'thermal'
for sub in ['thermal_maps', 'masks', 'rgb']:
    (td / sub).mkdir(parents=True, exist_ok=True)

ta = 0
for split in ['train', 'val']:
    rgb_d = c9 / 'Images' / '1_RGB' / split
    ir_d = c9 / 'Images' / '2_IR' / split
    mask_d = c9 / 'Images' / '0_Annotation' / split
    temp_d = c9 / 'Raw_Temperature_Data' / split
    if not rgb_d.exists():
        continue
    for rgb in rgb_d.glob('*.jpg'):
        stem = rgb.stem
        mask_f = mask_d / f'{stem}.png'
        if not mask_f.exists():
            continue
        shutil.copy2(rgb, td / 'rgb' / f'{stem}.jpg')
        mask = np.array(Image.open(mask_f))
        np.save(str(td / 'masks' / f'{stem}_mask.npy'), mask.astype(np.uint8))
        temp_csv = temp_d / f'{stem}.csv'
        ir_f = ir_d / f'{stem}.jpg'
        if temp_csv.exists():
            try:
                with open(temp_csv, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader)
                    rows = []
                    for row in reader:
                        vals = [float(v) for v in row[1:] if v.strip()]
                        if vals:
                            rows.append(vals)
                    if rows:
                        np.save(str(td / 'thermal_maps' / f'{stem}.npy'), np.array(rows, dtype=np.float32))
            except:
                if ir_f.exists():
                    ir = np.array(Image.open(ir_f).convert('L')).astype(np.float32)
                    np.save(str(td / 'thermal_maps' / f'{stem}.npy'), ir)
        elif ir_f.exists():
            ir = np.array(Image.open(ir_f).convert('L')).astype(np.float32)
            np.save(str(td / 'thermal_maps' / f'{stem}.npy'), ir)
        ta += 1
print(f'  thermal/ 추가: {ta}장 (RGB + 온도맵 + 세그마스크)')

# 5. room-interior (7,244장 polygon seg) → frames/
print('\n--- 5. room-interior → frames/ ---')
ri = RAW / 'M4_M5_M3' / 'room-interior'
fa = 0
for ss, sd in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
    di = DS / 'frames' / 'images' / sd
    dl = DS / 'frames' / 'labels' / sd
    di.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    si = ri / ss / 'images'
    sl = ri / ss / 'labels'
    if not si.exists():
        continue
    for img in si.glob('*.jpg'):
        lbl = sl / (img.stem + '.txt')
        if not lbl.exists():
            continue
        shutil.copy2(img, di / img.name)
        shutil.copy2(lbl, dl / lbl.name)
        fa += 1
print(f'  frames/ 추가: {fa}장')

# 6. S2DS (1,486장 세그마스크) → structural 보강
print('\n--- 6. S2DS → structural ---')
s2 = RAW / 'CUBIT-Seg(CUHK, 건물 외벽 특화),S2DS건물 구조 결함' / 's2ds'
sa = 0
for split in ['train', 'val', 'test']:
    sd = s2 / split
    if not sd.exists():
        continue
    imgs = [f for f in sd.glob('*.png') if '_lab' not in f.name]
    for img in imgs:
        di = DS / 'structural' / 'images' / split
        dl = DS / 'structural' / 'labels' / split
        shutil.copy2(img, di / f's2ds_{img.name}')
        with open(dl / f's2ds_{img.stem}.txt', 'w') as f:
            f.write('0 0.5 0.5 1.0 1.0\n')
        sa += 1
print(f'  structural 추가: {sa}장')

# 최종 카운트
print('\n' + '=' * 50)
print('  최종 datasets/ 카운트')
print('=' * 50)
for d in sorted(DS.iterdir()):
    if d.is_dir():
        count = sum(1 for _ in d.rglob('*') if _.is_file() and _.suffix in ('.jpg', '.png', '.npy'))
        print(f'  {d.name}: {count}')
