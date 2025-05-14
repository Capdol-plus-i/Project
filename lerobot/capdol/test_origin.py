import numpy as np

# ========== 테스트 전용 스크립트 ========== 
# 1) 파라미터 로드
param_data = np.load("model_parameters.npz")
parameters = {k: param_data[k] for k in param_data.files}

# 2) 데이터 로드 및 스케일링

X = np.array([[500], [150], [10]])
X = (X / 400).astype(np.float64)
#X = X.T

# 3) 순전파 함수 (ReLU 은닉, 선형 출력)
def L_model_forward(X, parameters):
    A = X
    L = len(parameters) // 2
    # 은닉층 (ReLU)
    for l in range(1, L):
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = W.dot(A) + b
        A = np.maximum(0, Z)
    # 출력층 (선형)
    Wl = parameters[f'W{L}']
    bl = parameters[f'b{L}']
    ZL = Wl.dot(A) + bl
    return ZL

# 4) 예측 및 출력
AL = L_model_forward(X, parameters)    # shape: (4, m)
pred = AL * 4000                     # 스케일 복원

# 샘플별 예측 결과 출력
print(f"샘플 예측 = {pred}")

# 모델 파라미터 재저장 (필요시)
save_flag = False
if save_flag:
    np.savez_compressed("model_parameters_tested", **parameters)
    print("Tested parameters saved.")