import time
import board
import neopixel_spi as neopixel
import speech_recognition as sr

# NeoPixel 설정 (SPI 방식)
NUM_PIXELS = 21
PIXEL_ORDER = neopixel.GRB
pixels = neopixel.NeoPixel_SPI(
    board.SPI(),
    NUM_PIXELS,
    pixel_order=PIXEL_ORDER,
    auto_write=False,
    brightness=1.0
)
pixels.fill((0, 0, 0))
pixels.show()

# 음성 인식 명령어
COMMANDS = ["불 켜", "불 꺼", "집중", "종료"]
r = sr.Recognizer()

print("음성 명령 대기 중...")

with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=1)

    while True:
        try:
            print("\n듣고 있어요...")
            audio = r.listen(source, timeout=5, phrase_time_limit=4)

            text = r.recognize_google(audio, language="ko-KR")
            print("인식:", text)

            if "불 켜" in text:
                print("조명을 켭니다.")
                pixels.fill((255, 255, 255))
                pixels.show()

            elif "불 꺼" in text:
                print("조명을 끕니다.")
                pixels.fill((0, 0, 0))
                pixels.show()

            elif "집중" in text:
                print("집중 모드: 밝은 흰색")
                pixels.fill((200, 200, 255))
                pixels.show()

            elif "종료" in text:
                print("종료합니다.")
                pixels.fill((0, 0, 0))
                pixels.show()
                break

            else:
                print("유효하지 않은 명령입니다. 다시 시도해주세요.")

        except sr.WaitTimeoutError:
            print("아무 말도 감지되지 않음.")
        except sr.UnknownValueError:
            print("말을 이해하지 못했습니다.")
        except sr.RequestError as e:
            print(f"구글 서버 오류: {e}")

