import numpy as np
import pandas as pd

def process_robot_hand_data(input_csv_path, output_npz_path):
    """
    로봇 핸드 데이터를 처리하는 함수입니다:
    1. CSV 파일을 읽습니다
    2. 필요한 7개 컬럼만 선택합니다:
       - cam1_tip_x, cam1_tip_y, cam2_tip_y, follower_joint1, follower_joint2, follower_joint3, follower_joint4
    3. 누락된 값이 있는 행을 제거합니다
    4. 데이터를 두 그룹으로 나눕니다:
       - 1~3번째 열: cam1_tip_y, cam2_tip_x, cam2_tip_y
       - 4~7번째 열: follower_joint1, follower_joint2, follower_joint3, follower_joint4
    5. 데이터를 NPZ 파일로 저장합니다
    
    Parameters:
    -----------
    input_csv_path : str
        입력 CSV 파일 경로
    output_npz_path : str
        출력 NPZ 파일 경로
    """
    print(f"CSV 파일 '{input_csv_path}'을(를) 처리합니다...")
    
    # 1. CSV 파일 읽기
    df = pd.read_csv(input_csv_path)
    
    print(f"원본 데이터 행 수: {len(df)}")
    print(f"원본 데이터 열: {df.columns.tolist()}")
    
    # 2. 필요한 7개 컬럼만 선택
    needed_columns = [
        'camera1_tip_x', 'camera1_tip_y', 'camera2_tip_y',
        'follower_joint_1', 'follower_joint_2', 'follower_joint_3', 'follower_joint_4'
    ]
    
    # 필요한 열이 모두 존재하는지 확인
    for col in needed_columns:
        if col not in df.columns:
            raise ValueError(f"필요한 열 '{col}'이(가) CSV 파일에 존재하지 않습니다.")
    
    # 필요한 열만 선택
    df_filtered = df[needed_columns]
    
    print(f"필요한 7개 열 선택 후 데이터 크기: {df_filtered.shape}")
    
    # 3. 누락된 값이 있는 행 제거
    df_clean = df_filtered.dropna()
    
    print(f"누락된 값 제거 후 행 수: {len(df_clean)}")
    
    # 4. 데이터를 두 그룹으로 나누기
    camera_data = df_clean.iloc[:, 0:3].to_numpy()    # 1~3번째 열
    joint_data = df_clean.iloc[:, 3:7].to_numpy()     # 4~7번째 열
    
    print(f"camera_data 배열 크기: {camera_data.shape}")
    print(f"joint_data 배열 크기: {joint_data.shape}")
    
    # 5. NPZ 파일로 저장
    np.savez_compressed(
        output_npz_path,
        camera_data=camera_data,  # 카메라 관련 데이터 (1-3번째 열)
        joint_data=joint_data     # 조인트 관련 데이터 (4-7번째 열)
    )
    
    print(f"데이터가 성공적으로 '{output_npz_path}'에 저장되었습니다.")
    
    # 데이터 샘플 확인
    print("\n처리된 데이터 샘플 (처음 5개 행):")
    print(df_clean.head())

# 메인 함수
if __name__ == "__main__":
    # 파일 경로 설정
    input_csv_path = "lerobot/capdol/collected_data/robot_hand_data.csv"  # 입력 CSV 파일 경로
    output_npz_path = "lerobot/capdol/robot_hand_data.npz"                # 출력 NPZ 파일 경로
    
    # 데이터 처리 및 저장
    process_robot_hand_data(input_csv_path, output_npz_path)
    
    # 저장된 데이터 확인 (선택 사항)
    print("\n저장된 데이터 확인:")
    data = np.load(output_npz_path)
    print(f"저장된 배열: {data.files}")
    print(f"camera_data shape: {data['camera_data'].shape}")
    print(f"joint_data shape: {data['joint_data'].shape}")
    
    # camera_data 첫 5개 행 출력
    print("\ncamera_data 첫 5개 행:")
    print(data['camera_data'][:5])
    
    # joint_data 첫 5개 행 출력
    print("\njoint_data 첫 5개 행:")
    print(data['joint_data'][:5])
    
    data.close()
    
    print("\n처리 완료!")