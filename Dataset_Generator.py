import os
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc

# === [실험할 때 여기만 바꾸면 됨!] ===
MAX_FRAMES = 30      # 윈도우 크기 (T)
TARGET_FPS = 3       # 목표 FPS
STRIDE = 10          # 윈도우 이동 간격
CONTEXT_MARGIN = 10  # 앞뒤 여유 프레임
CONFIDENCE_MASK = 0.5 # 신뢰도 마스킹 임계값

# === 경로 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, 'dataset')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'raw_data') # Step 1 결과물
SAVE_PATH = os.path.join(BASE_DIR, 'processed_dataset_v1') # 최종 저장소

CLASSES = ['Walking', 'Shopping', 'Fall', 'Threat']
LABEL_MAP = {
    'select': 'Shopping', 'compare': 'Shopping', 'inspect': 'Shopping',
    'purchase': 'Shopping', 'payment': 'Shopping', 'picking': 'Shopping',
    'fall': 'Falldown', 'faint': 'Falldown',
    'fight': 'Threat', 'assault': 'Threat', 'broken': 'Threat',
    'theft': 'Threat', 'vandalism': 'Threat',
}

def normalize_body_centric(sequence):
    """ 사람 중심 정규화 + 신뢰도 마스킹 """
    normalized_seq = np.copy(sequence)
    # (T, 17, 3)
    for t in range(sequence.shape[0]):
        frame = sequence[t]
        
        # 1. 신뢰도 마스킹 (노이즈 제거)
        low_conf = frame[:, 2] < CONFIDENCE_MASK
        frame[low_conf] = 0 # 좌표랑 신뢰도 다 0으로
        
        # 2. 중심 이동 (골반)
        if np.sum(frame[:, 2]) > 0: # 유효한 관절이 하나라도 있으면
            hip_x = (frame[11, 0] + frame[12, 0]) / 2
            hip_y = (frame[11, 1] + frame[12, 1]) / 2
            
            normalized_seq[t, :, 0] = frame[:, 0] - hip_x
            normalized_seq[t, :, 1] = frame[:, 1] - hip_y
            
            # 3. 스케일링
            width = np.max(frame[:, 0]) - np.min(frame[:, 0])
            height = np.max(frame[:, 1]) - np.min(frame[:, 1])
            scale = max(width, height, 1e-6)
            
            normalized_seq[t, :, 0] /= scale
            normalized_seq[t, :, 1] /= scale
            
            # 신뢰도 채널 유지
            normalized_seq[t, :, 2] = frame[:, 2]
        else:
             # 감지된 게 없으면 그냥 0으로 둠
            normalized_seq[t] = 0

    return normalized_seq

def process_single_file(file_info):
    raw_path, xml_path = file_info
    
    try:
        # 1. Raw 데이터 로드
        raw_dict = np.load(raw_path, allow_pickle=True).item()
        keypoints = raw_dict['keypoints'] # (Total_Frames, 17, 3)
        fps = raw_dict['fps']
        total_frames = len(keypoints)
        
        # 2. XML 파싱
        events = []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # (XML 파싱 로직은 이전과 동일하므로 생략하거나 함수로 분리 가능)
            # 여기서는 간략히 핵심만 구현
            starts, ends = {}, {}
            for track in root.findall('track'):
                label = track.get('label')
                if '_' not in label: continue
                act, time = label.rsplit('_', 1)
                act = act.strip().lower()
                if act not in LABEL_MAP: continue
                box = track.find('box')
                if box is None: continue
                fr = int(box.get('frame'))
                if time == 'start': starts.setdefault(act, []).append(fr)
                elif time == 'end': ends.setdefault(act, []).append(fr)
            
            for act in starts:
                if act in ends:
                    sl, el = sorted(starts[act]), sorted(ends[act])
                    for i in range(min(len(sl), len(el))):
                        events.append({'action': act, 'start': sl[i], 'end': el[i]})

        # 3. 샘플링 준비
        step = max(1, int(round(fps / TARGET_FPS)))
        # 실제 필요한 원본 프레임 수
        raw_window_size = MAX_FRAMES * step 
        stride_step = STRIDE * step
        
        samples = []
        labels = []
        
        # 라벨 마스크 생성
        label_array = np.zeros(total_frames, dtype=int) # 0: Walking
        for e in events:
            mapped = LABEL_MAP[e['action']]
            cls_idx = CLASSES.index(mapped)
            s = max(0, e['start'] - (CONTEXT_MARGIN * step))
            en = min(total_frames, e['end'] + (CONTEXT_MARGIN * step))
            label_array[s:en] = cls_idx # 덮어쓰기 (이상행동 우선)

        # 4. 슬라이딩 윈도우
        # 이벤트 구간 위주로 뽑되, Walking도 적절히 포함
        ptr = 0
        while ptr < total_frames:
            # 윈도우 범위
            start_idx = ptr
            end_idx = ptr + raw_window_size
            
            # 범위 벗어나면 Freeze 패딩 할거니까 일단 데이터 가져옴
            # 단, 시작점이 끝을 넘으면 종료
            if start_idx >= total_frames: break
            
            # 데이터 추출 (다운샘플링)
            # step 간격으로 가져옴. 범위 넘어가면 슬라이싱이 알아서 잘림 -> 나중에 패딩
            window_raw = keypoints[start_idx : end_idx : step]
            
            # 유효성 검사: 윈도우 내에 데이터가 너무 없으면(다 0이면) 스킵
            if np.sum(window_raw[:, :, 2]) == 0:
                ptr += stride_step
                continue

            # 라벨 결정 (윈도우 내 최빈값 or 이벤트 존재 여부)
            # 여기서는 윈도우 중간 지점의 라벨을 따르거나, Max Voting
            window_labels = label_array[start_idx : min(end_idx, total_frames) : step]
            if len(window_labels) > 0:
                # 이상 행동(Walking=0 아님)이 하나라도 있으면 그 라벨로
                abnormal = window_labels[window_labels > 0]
                if len(abnormal) > 0:
                    # 빈도수 높은 이상행동 선택
                    u, c = np.unique(abnormal, return_counts=True)
                    final_label = u[np.argmax(c)]
                else:
                    final_label = 0 # Walking
            else:
                final_label = 0
            
            # 5. 정규화 & 패딩
            norm_seq = normalize_body_centric(window_raw)
            
            # Freeze Padding
            curr_len = len(norm_seq)
            if curr_len < MAX_FRAMES:
                pad_amt = MAX_FRAMES - curr_len
                # Edge padding (Freeze)
                final_seq = np.pad(norm_seq, ((0, pad_amt), (0,0), (0,0)), mode='edge')
            else:
                final_seq = norm_seq[:MAX_FRAMES]
            
            # (T, V, C) -> (C, T, V, M=1)
            # (30, 17, 3) -> (3, 30, 17, 1)
            final_data = np.expand_dims(final_seq.transpose(2, 0, 1), axis=-1)
            
            samples.append(final_data)
            labels.append(final_label)
            
            ptr += stride_step

        return samples, labels

    except Exception as e:
        print(f"Error {raw_path}: {e}")
        return [], []

def main():
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 처리할 파일 목록
    raw_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.npy')]
    task_list = []
    
    print(f"🚀 Step 2 시작: {len(raw_files)}개 Raw 파일 가공")
    
    for f in raw_files:
        raw_path = os.path.join(RAW_DATA_PATH, f)
        # XML 파일은 dataset 폴더 구조를 따라가야 함. 
        # (파일명이 유니크하다면 os.walk로 찾거나, 경로 규칙에 따라 매핑)
        # 여기서는 간단히 raw_data 생성시 파일명을 유지했다고 가정하고 dataset 폴더에서 찾음
        base_name = os.path.splitext(f)[0]
        # 원본 XML 찾기 (재귀 검색)
        xml_path = None
        for r, d, files in os.walk(DATASET_ROOT):
            if base_name + '.xml' in files:
                xml_path = os.path.join(r, base_name + '.xml')
                break
        
        if xml_path:
            task_list.append((raw_path, xml_path))
    
    # 병렬 처리 (CPU 풀가동)
    all_X = []
    all_Y = []
    
    with Pool(cpu_count()) as pool:
        for X, Y in tqdm(pool.imap_unordered(process_single_file, task_list), total=len(task_list)):
            if X:
                all_X.extend(X)
                all_Y.extend(Y)
                
    # 최종 저장
    print("💾 병합 및 저장 중...")
    X_final = np.array(all_X, dtype=np.float32)
    Y_final = np.array(all_Y, dtype=np.int64)
    
    np.save(os.path.join(SAVE_PATH, 'train_data.npy'), X_final)
    np.save(os.path.join(SAVE_PATH, 'train_label.npy'), Y_final)
    
    print(f"끝! 데이터 Shape: {X_final.shape}")
    
if __name__ == "__main__":
    main()