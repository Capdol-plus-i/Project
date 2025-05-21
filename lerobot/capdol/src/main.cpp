#include <Arduino.h>
#include <TaskScheduler.h>
#include <PinChangeInterrupt.h>
#include <Adafruit_NeoPixel.h>

// 핀 및 상수 정의
const int LED_BUTTON_PIN = 2;     // 밝기 조절 버튼
const int ROBOT_BUTTON_PIN = 3;   // 로봇팔 제어 버튼
const int NEOPIXEL_PIN = 6;
const int NUM_PIXELS = 21;
const int MAX_BRIGHTNESS = 255;
const int BRIGHTNESS_STEPS = 5; 

volatile int brightnessLevel = 0;   // 현재 밝기 단계 (0~5)
volatile bool ledButtonPressed = false;
volatile bool robotButtonPressed = false;
volatile int robotMode = 0;         // 로봇 모드 (0 또는 1)
int currentBrightness = 0;

// Neopixel 설정
Adafruit_NeoPixel pixels(NUM_PIXELS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

// 태스크 스케줄러
Scheduler runner;
void displayStatusTask();
void processButtonsTask();
void updateNeopixelTask();

// 태스크 정의
Task tDisplayStatus(1000, TASK_FOREVER, &displayStatusTask);
Task tProcessButtons(50, TASK_FOREVER, &processButtonsTask);
Task tUpdateNeopixel(100, TASK_FOREVER, &updateNeopixelTask);

// 버튼 인터럽트 핸들러
void ledButtonInterrupt() {
  ledButtonPressed = true;
}

void robotButtonInterrupt() {
  robotButtonPressed = true;
}

void setup() {
  Serial.begin(9600);
  Serial.println("Neopixel 밝기 조절 및 로봇 제어 프로그램 시작");

  pinMode(LED_BUTTON_PIN, INPUT_PULLUP);
  pinMode(ROBOT_BUTTON_PIN, INPUT_PULLUP);
  
  attachPCINT(digitalPinToPCINT(LED_BUTTON_PIN), ledButtonInterrupt, FALLING);
  attachPCINT(digitalPinToPCINT(ROBOT_BUTTON_PIN), robotButtonInterrupt, FALLING);

  pixels.begin();
  pixels.clear();
  pixels.show();

  runner.init();
  runner.addTask(tDisplayStatus);
  runner.addTask(tProcessButtons);
  runner.addTask(tUpdateNeopixel);
  tDisplayStatus.enable();
  tProcessButtons.enable();
  tUpdateNeopixel.enable();

  Serial.println("초기 상태: 밝기 단계 0, 로봇 모드 0");
}

void displayStatusTask() {
  Serial.print("밝기 단계: ");
  Serial.print(brightnessLevel);
  Serial.print(", 밝기: ");
  Serial.print(currentBrightness);
  Serial.print(", 로봇 모드: ");
  Serial.println(robotMode);
}

void processButtonsTask() {
  // LED 밝기 조절 버튼 처리
  if (ledButtonPressed) {
    brightnessLevel = (brightnessLevel + 1) % (BRIGHTNESS_STEPS + 1); // 0~5 반복
    currentBrightness = brightnessLevel * (MAX_BRIGHTNESS / BRIGHTNESS_STEPS);
    ledButtonPressed = false;

    Serial.print("밝기 버튼! 밝기 단계: ");
    Serial.print(brightnessLevel);
    Serial.print(", 새 밝기: ");
    Serial.println(currentBrightness);
    
    // LED 밝기 명령 전송
    Serial.print("CMD:LED:");
    Serial.println(brightnessLevel);
  }
  
  // 로봇 제어 버튼 처리
  if (robotButtonPressed) {
    robotMode = !robotMode;  // 0과 1 사이 토글
    robotButtonPressed = false;
    
    Serial.print("로봇 버튼! 로봇 모드: ");
    Serial.println(robotMode);
    
    // 로봇 제어 명령 전송
    Serial.print("CMD:ROBOT:");
    Serial.println(robotMode);
  }
}

void updateNeopixelTask() {
  for (int i = 0; i < NUM_PIXELS; i++) {
    // 흰색 (255, 255, 255) 밝기만 조절
    pixels.setPixelColor(i, pixels.Color(currentBrightness, currentBrightness, currentBrightness));
  }
  pixels.show();
}

void loop() {
  runner.execute();
}