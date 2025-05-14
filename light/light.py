import time
import board
import neopixel_spi as neopixel

NUM_PIXELS = 21
PIXEL_ORDER = neopixel.GRB

BRIGHTNESS_LEVELS = {
    "1": 0.2,
    "2": 0.4,
    "3": 0.6,
    "4": 0.8,
    "5": 1.0
}

COLOR_TEMPS = {
    "c": (160, 180, 255),  # 차가운 흰색 (파란색 강조)
    "n": (255, 255, 255),  # 중립 흰색
    "w": (255, 180, 100)   # 따뜻한 흰색 (붉은색 강조)
}


current_color = COLOR_TEMPS["n"]  # 기본 중립색

spi = board.SPI()

pixels = neopixel.NeoPixel_SPI(
    spi,
    NUM_PIXELS,
    pixel_order=PIXEL_ORDER,
    auto_write=False,
    brightness=0.2
)

def set_brightness(level_str):
    if level_str in BRIGHTNESS_LEVELS:
        brightness = BRIGHTNESS_LEVELS[level_str]
        pixels.brightness = brightness
        pixels.fill(current_color)
        pixels.show()
        print(f"밝기 {level_str}단계로 설정됨 (밝기값: {brightness})")
    else:
        print("잘못된 입력입니다. 1~5 사이 숫자를 입력하세요.")

def set_color_temp(mode):
    global current_color
    if mode in COLOR_TEMPS:
        current_color = COLOR_TEMPS[mode]
        pixels.fill(current_color)
        pixels.show()
        label = {"c": "차가운 색", "n": "중립 색", "w": "따뜻한 색"}[mode]
        print(f"{label}으로 설정됨 (RGB: {current_color})")
    else:
        print("잘못된 색온도 입력입니다. c(차가움), n(중립), w(따뜻함) 중 선택하세요.")

# 메인 루프
print("밝기: 1~5 | 색온도: c(차가움), n(중립), w(따뜻함) | 종료: q")

while True:
    user_input = input("입력: ").strip().lower()
    if user_input == "q":
        print("종료합니다.")
        pixels.fill((0, 0, 0))  # LED 끄기
        pixels.show()
        break
    elif user_input in BRIGHTNESS_LEVELS:
        set_brightness(user_input)
    elif user_input in COLOR_TEMPS:
        set_color_temp(user_input)
    else:
        print("잘못된 입력입니다.")

