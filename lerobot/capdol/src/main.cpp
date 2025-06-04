#include <Arduino.h>
#include <TaskScheduler.h>
#include <PinChangeInterrupt.h>
#include <Adafruit_NeoPixel.h>

// 핀 및 상수 정의
const int LED_BUTTON_PIN = 2;     // 밝기 조절 버튼
const int ROBOT_BUTTON_PIN = 3;   // 로봇팔 제어 버튼
const int NEOPIXEL_PIN = 4;
const int NUM_PIXELS = 16;
const int MAX_BRIGHTNESS = 255;
const int BRIGHTNESS_STEPS = 5; 
const int DEBOUNCE_DELAY = 200;   // 디바운싱 딜레이 (ms)

// 상태 변수
volatile int brightnessLevel = 0;   // 현재 밝기 단계 (0~5)
volatile bool ledButtonPressed = false;
volatile bool robotButtonPressed = false;
volatile unsigned long lastLedButtonTime = 0;
volatile unsigned long lastRobotButtonTime = 0;
volatile int robotMode = 0;         // 로봇 모드 (0 또는 1)
int currentBrightness = 0;
bool systemConnected = false;       // Python 시스템 연결 상태
unsigned long lastHeartbeat = 0;
unsigned long lastStatusSend = 0;
const unsigned long HEARTBEAT_INTERVAL = 5000;  // 5초마다 상태 전송
const unsigned long STATUS_INTERVAL = 1000;     // 1초마다 상태 출력

// LED 효과 변수
bool ledEffectActive = false;
unsigned long ledEffectStart = 0;
int effectType = 0; // 0: 없음, 1: 깜빡임, 2: 페이드, 3: 무지개

// Neopixel 설정
Adafruit_NeoPixel pixels(NUM_PIXELS, NEOPIXEL_PIN, NEO_RGBW + NEO_KHZ800);

// 함수 프로토타입 선언
void displayStatusTask();
void processButtonsTask();
void updateNeopixelTask();
void processSerialTask();
void heartbeatTask();
void ledButtonInterrupt();
void robotButtonInterrupt();
void startupEffect();
void sendStatus();
void triggerLedEffect(int type);
void setAllPixels(int r, int g, int b);
uint32_t Wheel(byte WheelPos);

// 태스크 스케줄러
Scheduler runner;

// 태스크 정의
Task tDisplayStatus(STATUS_INTERVAL, TASK_FOREVER, &displayStatusTask);
Task tProcessButtons(50, TASK_FOREVER, &processButtonsTask);
Task tUpdateNeopixel(50, TASK_FOREVER, &updateNeopixelTask);  // 더 부드러운 LED 효과를 위해 50ms로 단축
Task tProcessSerial(100, TASK_FOREVER, &processSerialTask);
Task tHeartbeat(HEARTBEAT_INTERVAL, TASK_FOREVER, &heartbeatTask);

// 버튼 인터럽트 핸들러 (디바운싱 포함)
void ledButtonInterrupt() {
  unsigned long currentTime = millis();
  if (currentTime - lastLedButtonTime > DEBOUNCE_DELAY) {
    ledButtonPressed = true;
    lastLedButtonTime = currentTime;
  }
}

void robotButtonInterrupt() {
  unsigned long currentTime = millis();
  if (currentTime - lastRobotButtonTime > DEBOUNCE_DELAY) {
    robotButtonPressed = true;
    lastRobotButtonTime = currentTime;
  }
}

void setup() {
  Serial.begin(9600);
  Serial.println("=== Robot Control Arduino Module Started ===");
  Serial.println("Version: 1.0");
  Serial.println("Compatible with Python Robot Control System");

  pinMode(LED_BUTTON_PIN, INPUT_PULLUP);
  pinMode(ROBOT_BUTTON_PIN, INPUT_PULLUP);
  
  attachPCINT(digitalPinToPCINT(LED_BUTTON_PIN), ledButtonInterrupt, FALLING);
  attachPCINT(digitalPinToPCINT(ROBOT_BUTTON_PIN), robotButtonInterrupt, FALLING);

  pixels.begin();
  pixels.clear();
  pixels.show();

  // 시작 시 LED 효과
  startupEffect();

  runner.init();
  runner.addTask(tDisplayStatus);
  runner.addTask(tProcessButtons);
  runner.addTask(tUpdateNeopixel);
  runner.addTask(tProcessSerial);
  runner.addTask(tHeartbeat);
  
  tDisplayStatus.enable();
  tProcessButtons.enable();
  tUpdateNeopixel.enable();
  tProcessSerial.enable();
  tHeartbeat.enable();

  // 초기 상태 전송
  sendStatus();
  Serial.println("Arduino Ready - Waiting for Python system connection...");
}

void startupEffect() {
  // 시작 시 무지개 효과
  for(int j = 0; j < 256; j++) {
    for(int i = 0; i < NUM_PIXELS; i++) {
      pixels.setPixelColor(i, Wheel((i * 256 / NUM_PIXELS + j) & 255));
    }
    pixels.show();
    delay(5);
  }
  pixels.clear();
  pixels.show();
}

void displayStatusTask() {
  if (millis() - lastStatusSend > STATUS_INTERVAL) {
    Serial.print("Status - Brightness: ");
    Serial.print(brightnessLevel);
    Serial.print("/");
    Serial.print(BRIGHTNESS_STEPS);
    Serial.print(" (");
    Serial.print(currentBrightness);
    Serial.print("), Robot Mode: ");
    Serial.print(robotMode);
    Serial.print(", Connected: ");
    Serial.print(systemConnected ? "YES" : "NO");
    Serial.print(", Effect: ");
    Serial.println(effectType);
    lastStatusSend = millis();
  }
}

void processButtonsTask() {
  // LED 밝기 조절 버튼 처리
  if (ledButtonPressed) {
    brightnessLevel = (brightnessLevel + 1) % (BRIGHTNESS_STEPS + 1); // 0~5 반복
    currentBrightness = brightnessLevel * (MAX_BRIGHTNESS / BRIGHTNESS_STEPS);
    ledButtonPressed = false;

    Serial.print("LED Button Pressed! Brightness Level: ");
    Serial.print(brightnessLevel);
    Serial.print(" -> Brightness: ");
    Serial.println(currentBrightness);
    
    // LED 밝기 명령 전송 (Python 시스템이 필요하다면)
    Serial.print("CMD:LED:");
    Serial.println(brightnessLevel);
    
    // LED 효과 트리거
    triggerLedEffect(1); // 깜빡임 효과
    sendStatus();
  }
  
  // 로봇 제어 버튼 처리
  if (robotButtonPressed) {
    robotMode = !robotMode;  // 0과 1 사이 토글
    robotButtonPressed = false;
    
    Serial.print("Robot Button Pressed! Mode changed to: ");
    Serial.println(robotMode);
    
    // 로봇 제어 명령 전송 (Python 시스템에서 수신)
    Serial.print("CMD:ROBOT:");
    Serial.println(robotMode);
    
    // LED 효과 트리거
    triggerLedEffect(robotMode == 0 ? 2 : 3); // 모드에 따른 다른 효과
    sendStatus();
  }
}

void processSerialTask() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("HEARTBEAT")) {
      systemConnected = true;
      lastHeartbeat = millis();
      Serial.println("ACK:HEARTBEAT");
    }
    else if (command.startsWith("STATUS")) {
      sendStatus();
    }
    else if (command.startsWith("SET_BRIGHTNESS:")) {
      int newLevel = command.substring(15).toInt();
      if (newLevel >= 0 && newLevel <= BRIGHTNESS_STEPS) {
        brightnessLevel = newLevel;
        currentBrightness = brightnessLevel * (MAX_BRIGHTNESS / BRIGHTNESS_STEPS);
        Serial.print("Brightness set to level: ");
        Serial.println(brightnessLevel);
        triggerLedEffect(1);
      }
    }
    else if (command.startsWith("SET_MODE:")) {
      int newMode = command.substring(9).toInt();
      if (newMode == 0 || newMode == 1) {
        robotMode = newMode;
        Serial.print("Robot mode set to: ");
        Serial.println(robotMode);
        triggerLedEffect(newMode == 0 ? 2 : 3);
      }
    }
    else if (command.startsWith("LED_EFFECT:")) {
      int effect = command.substring(11).toInt();
      triggerLedEffect(effect);
    }
    else if (command == "RESET") {
      // 시스템 리셋
      brightnessLevel = 0;
      currentBrightness = 0;
      robotMode = 0;
      triggerLedEffect(0);
      Serial.println("System Reset");
      sendStatus();
    }
    else {
      Serial.print("Unknown command: ");
      Serial.println(command);
    }
  }
  
  // 연결 상태 확인 (5초 이상 heartbeat 없으면 연결 끊김으로 판단)
  if (systemConnected && (millis() - lastHeartbeat > HEARTBEAT_INTERVAL * 2)) {
    systemConnected = false;
    Serial.println("Warning: Python system connection lost");
  }
}

void heartbeatTask() {
  // 주기적으로 상태 전송
  sendStatus();
}

void sendStatus() {
  Serial.print("STATUS:BRIGHTNESS:");
  Serial.print(brightnessLevel);
  Serial.print(":MODE:");
  Serial.print(robotMode);
  Serial.print(":CONNECTED:");
  Serial.println(systemConnected ? "1" : "0");
}

void triggerLedEffect(int type) {
  effectType = type;
  ledEffectActive = true;
  ledEffectStart = millis();
}

void updateNeopixelTask() {
  if (ledEffectActive) {
    unsigned long elapsed = millis() - ledEffectStart;
    
    switch(effectType) {
      case 1: // 깜빡임 효과 (밝기 변경 시)
        if (elapsed < 100) {
          int flashBrightness = (elapsed % 100 < 50) ? currentBrightness : 0;
          setAllPixels(flashBrightness, flashBrightness, flashBrightness);
        } else {
          ledEffectActive = false;
        }
        break;
        
      case 2: // 파란색 페이드 (모드 0)
        if (elapsed < 1000) {
          int intensity = (sin((elapsed / 1000.0) * PI * 2) + 1) * currentBrightness / 2;
          setAllPixels(0, 0, intensity);
        } else {
          ledEffectActive = false;
        }
        break;
        
      case 3: // 초록색 페이드 (모드 1)
        if (elapsed < 1000) {
          int intensity = (sin((elapsed / 1000.0) * PI * 2) + 1) * currentBrightness / 2;
          setAllPixels(0, intensity, 0);
        } else {
          ledEffectActive = false;
        }
        break;
        
      default:
        ledEffectActive = false;
        break;
    }
  } else {
    // 일반 상태 - 흰색 LED
    setAllPixels(currentBrightness, currentBrightness, currentBrightness);
  }
  
  // 연결 상태 표시 (첫 번째 LED)
  if (!systemConnected && NUM_PIXELS > 0) {
    // 연결 안됨 - 빨간색 깜빡임
    int redIntensity = (millis() % 1000 < 500) ? 50 : 0;
    pixels.setPixelColor(0, pixels.Color(redIntensity, 0, 0));
  }
  
  pixels.show();
}

void setAllPixels(int r, int g, int b) {
  for (int i = 0; i < NUM_PIXELS; i++) {
    pixels.setPixelColor(i, pixels.Color(r, g, b));
  }
}

// 무지개 색상 생성 함수
uint32_t Wheel(byte WheelPos) {
  WheelPos = 255 - WheelPos;
  if(WheelPos < 85) {
    return pixels.Color(255 - WheelPos * 3, 0, WheelPos * 3);
  }
  if(WheelPos < 170) {
    WheelPos -= 85;
    return pixels.Color(0, WheelPos * 3, 255 - WheelPos * 3);
  }
  WheelPos -= 170;
  return pixels.Color(WheelPos * 3, 255 - WheelPos * 3, 0);
}

void loop() {
  runner.execute();
}