import speech_recognition as sr

WAKE_WORD = "더미"
COMMANDS = ["왼쪽", "오른쪽", "위", "아래", "종료"]
r = sr.Recognizer()

def clean_text(text):
    """인식된 텍스트에서 공백과 특수문자를 제거"""
    return text.replace(" ", "").strip()

with sr.Microphone() as source:
    # 소음 기준 조정 (수동 threshold 조정)
    print("마이크 초기화 중...")
    r.adjust_for_ambient_noise(source, duration=1)
    r.energy_threshold = 400
    print("웨이크워드 대기 중...")

    while True:
        try:
            print("\n 웨이크워드 듣는 중...")
            wake_audio = r.listen(source, timeout=5, phrase_time_limit=4)
            wake_text = r.recognize_google(wake_audio, language="ko-KR")
            wake_text_clean = clean_text(wake_text)
            print(f"인식된 텍스트: '{wake_text_clean}'")

            if WAKE_WORD == wake_text_clean:
                print("웨이크워드 인식됨! 명령어를 말해주세요.")
                
                try:
                    command_audio = r.listen(source, timeout=5, phrase_time_limit=5)
                    command_text = r.recognize_google(command_audio, language="ko-KR")
                    command_text_clean = clean_text(command_text)
                    print(f"명령어 인식: '{command_text_clean}'")

                    recognized = False
                    for cmd in COMMANDS:
                        if cmd == command_text_clean:
                            print(f"명령 인식: '{cmd}'")
                            recognized = True

                            # 🛠 명령 실행
                            if cmd == "왼쪽":
                                print("← 왼쪽으로 이동!")
                            elif cmd == "오른쪽":
                                print("→ 오른쪽으로 이동!")
                            elif cmd == "위로":
                                print("↑ 위로 이동!")
                            elif cmd == "아래로":
                                print("↓ 아래로 이동!")
                            elif cmd == "종료":
                                print("🛑 시스템 종료")
                                exit(0)
                            break

                    if not recognized:
                        print("유효하지 않은 명령입니다. 다시 시도해주세요.")
                except sr.UnknownValueError:
                    print("명령어 인식을 실패했습니다. 다시 시도해주세요.")
                except sr.RequestError as e:
                    print(f"구글 서버 오류: {e}")

            else:
                print("웨이크워드가 아닙니다. 다시 듣습니다.")

        except sr.WaitTimeoutError:
            print("아무 말도 감지되지 않음. 다시 듣습니다.")
        except sr.UnknownValueError:
            print("말을 이해하지 못했습니다. 다시 시도해주세요.")
        except sr.RequestError as e:
            print(f"구글 서버 오류: {e}")