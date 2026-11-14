// Fridge Guardian v1 - MCU side (STM32 on the Arduino UNO Q)
//
// Reads the sensors in real time and drives the outputs. All AI runs on the
// Linux side (python/main.py); this sketch only moves data over the Bridge.
//
// Sensors (all optional, the sketch keeps running if one is missing):
//   Modulino Thermo    -> temperature, humidity          -> "sensors" (every 1 s)
//   Modulino Light     -> lux (door open detection)      -> "sensors"
//   Modulino Buttons   -> A = put in, B = take out, C = silence -> "button"
// Outputs (called from Python):
//   "set_pixels"(r, g, b)       Modulino Pixels status light
//   "beep"(freq_hz, ms)         Modulino Buzzer
//   "draw"(frame[104])          built-in 8x13 LED matrix, values 0..7

#include <Arduino_RouterBridge.h>
#include <Arduino_Modulino.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

// ---- Configuration ---------------------------------------------------------
const unsigned long SENSOR_INTERVAL_MS = 1000;   // send sensor readings once per second

// ---- Hardware objects ------------------------------------------------------
ModulinoThermo thermo;
ModulinoLight light;
ModulinoButtons buttons;
ModulinoPixels pixels;
ModulinoBuzzer buzzer;
Arduino_LED_Matrix matrix;

bool hasThermo = false, hasLight = false;
bool hasButtons = false, hasPixels = false, hasBuzzer = false;

const uint8_t FRAME_SIZE = 8 * 13;
uint8_t frame[FRAME_SIZE] = {0};

unsigned long lastSensors = 0;
unsigned long lastRetry = 0;

// ---- Functions Python can call ---------------------------------------------
void set_pixels(int r, int g, int b) {
  if (!hasPixels) return;
  for (int i = 0; i < 8; i++) {
    pixels.set(i, (uint8_t)r, (uint8_t)g, (uint8_t)b, 10);
  }
  pixels.show();
}

void beep(int freq, int ms) {
  if (!hasBuzzer) return;
  buzzer.tone((size_t)freq, (size_t)ms);
}

void draw(std::vector<uint8_t> newFrame) {
  size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
  memcpy(frame, newFrame.data(), len);
}

// ---- Setup -----------------------------------------------------------------
void beginModules() {
  if (!hasThermo) hasThermo = thermo.begin();
  if (!hasLight) hasLight = light.begin();
  if (!hasButtons) hasButtons = buttons.begin();
  if (!hasPixels) hasPixels = pixels.begin();
  if (!hasBuzzer) hasBuzzer = buzzer.begin();
  if (hasButtons) buttons.setLeds(true, true, true);
}

void setup() {
  Bridge.begin();
  Modulino.begin(Wire1);   // Qwiic connector on the UNO Q
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();

  beginModules();

  Bridge.provide("set_pixels", set_pixels);
  Bridge.provide("beep", beep);
  Bridge.provide("draw", draw);
}

// ---- Loop ------------------------------------------------------------------
void loop() {
  unsigned long now = millis();
  matrix.draw(frame);

  // Retry missing modules every 5 s, so they can be plugged in while running
  if (now - lastRetry >= 5000) {
    lastRetry = now;
    beginModules();
  }

  // Buttons: A = put in, B = take out, C = silence alarm
  if (hasButtons && buttons.update()) {
    for (int i = 0; i < 3; i++) {
      if (buttons.isPressed(i)) Bridge.notify("button", i);
    }
  }

  // Environment and door light, once per second
  if (now - lastSensors >= SENSOR_INTERVAL_MS) {
    lastSensors = now;
    float t = hasThermo ? thermo.getTemperature() : -999.0;
    float h = hasThermo ? thermo.getHumidity() : -999.0;
    float lux = -1.0;
    if (hasLight && light.update()) lux = (float)light.getLux();
    Bridge.notify("sensors", t, h, lux);
  }

  delay(1);
}
