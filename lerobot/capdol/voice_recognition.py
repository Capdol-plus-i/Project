import pyaudio
import time
from six.moves import queue
from google.cloud import speech

RATE = 16000
CHUNK = int(RATE / 5)  # 100ms 단위

WAKE_WORDS = ["하이봇", "하이못", "아이봇", "AI봇", "아이", "하이"]
CMD_MAP = {
    "왼쪽": ["왼쪽", "왼 쪽", "왼"],
    "오른쪽": ["오른쪽", "오른 쪽", "오른","5"],
    "위": ["위", "위로", "위쪽"],
    "아래": ["아래", "아래로", "아레"],
    "종료": ["종료", "끝내", "종료해"]
}


class MicrophoneStream:
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._stream.stop_stream()
        self._stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)


def start_stream(is_command_mode=False):
    client = speech.SpeechClient()

    # 인식 우선순위 키워드 등록
    speech_context = speech.SpeechContext(
        phrases=WAKE_WORDS + sum(CMD_MAP.values(), []),
        boost=20.0
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ko-KR",
        speech_contexts=[speech_context],
        model="command_and_search"
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=is_command_mode  # 명령어 모드일 때는 말 끝나면 자동 종료
    )

    stream = MicrophoneStream(RATE, CHUNK)
    stream.__enter__()
    audio_generator = stream.generator()
    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
    responses = client.streaming_recognize(streaming_config, requests)
    return stream, responses


def main():
    print("🎙️ 시스템이 준비되었습니다. 웨이크워드를 말해주세요 (예: '하이봇')")

    while True:
        stream, responses = start_stream()

        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript.strip()
            print(" 인식된 내용 :", transcript)

            if any(wake in transcript for wake in WAKE_WORDS):
                if result.is_final or result.stability > 0.8:
                    print(" 웨이크워드 감지됨 명령을 말씀하세요.")
                    stream.__exit__(None, None, None)
                    time.sleep(0.1)
                    # 명령어 모드 진입
                    command_stream, command_responses = start_stream(is_command_mode=True)
                    time.sleep(0.1)  # 스트림 안정화 대기
                    start_time = time.time()
                    MAX_COMMAND_DURATION = 5  # 초과 시 자동 종료

                    for cmd_response in command_responses:
                        if time.time() - start_time > MAX_COMMAND_DURATION:
                            print(" 명령어 대기 시간 초과")
                            command_stream.__exit__(None, None, None)
                            break

                        if not cmd_response.results:
                            continue
                        cmd_result = cmd_response.results[0]
                        if not cmd_result.alternatives:
                            continue

                        cmd_transcript = cmd_result.alternatives[0].transcript.strip()
                        print(" 말한 내용 :", cmd_transcript)

                        for cmd, variations in CMD_MAP.items():
                            if any(v in cmd_transcript for v in variations):
                                print(f" 명령어: {cmd}")
                                if cmd == "종료":
                                    print(" 시스템 종료")
                                    command_stream.__exit__(None, None, None)
                                    return
                                elif cmd == "왼쪽":
                                    print(" 왼쪽으로 이동")
                                elif cmd == "오른쪽":
                                    print(" 오른쪽으로 이동")
                                elif cmd == "위":
                                    print(" 위로 이동")
                                elif cmd == "아래":
                                    print(" 아래로 이동")

                                command_stream.__exit__(None, None, None)
                                break
                        break
                    break


if __name__ == "__main__":
    main()