#include <Arduino.h>
#include <TaskScheduler.h>
#include <PinChangeInterrupt.h>
#include <Adafruit_NeoPixel.h>

// 상수 정의
const int BUTTON_PIN = 2;       // 버튼이 연결된 핀
const int NEOPIXEL_PIN = 6;     // Neopixel이 연결된 핀
const int NUM_PIXELS = 8;       // Neopixel LED 개수
const int MAX_BRIGHTNESS = 255; // 최대 밝기
const int BRIGHTNESS_STEPS = 5; // 밝기 단계(0~255를 5단계로 나눔)

volatile int counter = 0;             // 카운터 변수
volatile bool buttonPressed = false;  // 버튼 눌림 상태 플래그
int currentBrightness = 0;            // 현재 밝기

// Neopixel 설정
Adafruit_NeoPixel pixels(NUM_PIXELS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

// 태스크 스케줄러 설정
Scheduler runner;

// 태스크 함수 선언
void displayStatusTask();
void processButtonTask();
void updateNeopixelTask();

// 태스크 정의
Task tDisplayStatus(1000, TASK_FOREVER, &displayStatusTask);
Task tProcessButton(50, TASK_FOREVER, &processButtonTask);
Task tUpdateNeopixel(100, TASK_FOREVER, &updateNeopixelTask);

// 버튼 인터럽트 핸들러
void buttonInterrupt() {
  buttonPressed = true;
}

void setup() {
  // 시리얼 초기화
  Serial.begin(9600);
  Serial.println("Neopixel 밝기 조절 프로그램 시작");
  
  // 버튼 핀 설정
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // PinChangeInterrupt 설정
  attachPCINT(digitalPinToPCINT(BUTTON_PIN), buttonInterrupt, FALLING);
  
  // Neopixel 초기화
  pixels.begin();
  pixels.clear();
  pixels.show();
  
  // 태스크 시작
  runner.init();
  runner.addTask(tDisplayStatus);
  runner.addTask(tProcessButton);
  runner.addTask(tUpdateNeopixel);
  tDisplayStatus.enable();
  tProcessButton.enable();
  tUpdateNeopixel.enable();
  
  // 초기 상태 표시
  Serial.println("초기 상태: 밝기 0");
}

// 상태 표시 태스크 함수
void displayStatusTask() {
  Serial.print("카운트: ");
  Serial.print(counter);
  Serial.print(", 밝기: ");
  Serial.println(currentBrightness);
}

// 버튼 처리 태스크 함수
void processButtonTask() {
  if (buttonPressed) {
    // 버튼이 눌렸으면 카운터 증가
    counter++;
    
    // 밝기 단계 계산 (0 ~ MAX_BRIGHTNESS 범위에서 BRIGHTNESS_STEPS 단계로)
    currentBrightness = (counter % (BRIGHTNESS_STEPS + 1)) * (MAX_BRIGHTNESS / BRIGHTNESS_STEPS);
    
    buttonPressed = false;
    
    // 상태 변경 표시
    Serial.print("버튼 눌림! 카운트: ");
    Serial.print(counter);
    Serial.print(", 새 밝기: ");
    Serial.println(currentBrightness);
  }
}

// Neopixel 업데이트 태스크 함수
void updateNeopixelTask() {
  // 모든 LED에 같은 색상과 밝기 설정 (파란색 사용)
  for (int i = 0; i < NUM_PIXELS; i++) {
    // 색상은 파란색(0,0,255)으로 고정, 밝기만 변경
    pixels.setPixelColor(i, pixels.Color(0, 0, currentBrightness));
  }
  pixels.show();
}

void loop() {
  // 태스크 실행
  runner.execute();
}