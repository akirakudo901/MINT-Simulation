/*
 * ReadAnglesFromFile - reads hip,knee,ankle triplets from random_angles.txt on SD card
 * and applies them to the servos.
 *
 * HARDWARE: Requires SD card module (SPI). Typical wiring for Arduino Uno:
 *   SD Module -> Arduino
 *   CS   -> pin 4
 *   MOSI -> pin 11
 *   MISO -> pin 12
 *   SCK  -> pin 13
 *   VCC  -> 5V, GND -> GND
 *
 * SETUP: Copy random_angles.txt to the root of a FAT-formatted SD card,
 *        insert the card into the module, then upload and run.
 */

#include <SPI.h>
#include <SD.h>
#include <Servo.h>

Servo hipL;
Servo hipR;
Servo kneeL;
Servo kneeR;
Servo ankleL;
Servo ankleR;

// SD card
#define SD_CS_PIN 4
#define FILENAME "random_angles.txt"

// Joint limits (clamp values from file)
#define HIP_MIN 0
#define HIP_MAX 180
#define KNEE_MIN 0
#define KNEE_MAX 150
#define ANKLE_MIN 0
#define ANKLE_MAX 120

void setup() {
  Serial.begin(9600);

  hipL.attach(10);
  kneeL.attach(9);
  ankleL.attach(8);

  hipL.write(0);
  kneeL.write(0);
  ankleL.write(0);
  delay(1000);

  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("SD card init failed. Check wiring and card.");
    return;
  }
  Serial.println("SD card OK.");

  if (!SD.exists(FILENAME)) {
    Serial.print("File not found: ");
    Serial.println(FILENAME);
    return;
  }

  File f = SD.open(FILENAME, FILE_READ);
  if (!f) {
    Serial.println("Could not open file for reading.");
    return;
  }

  int count = 0;
  while (f.available()) {
    String line = f.readStringUntil('\n');
    line.trim();

    if (line.length() == 0) continue;

    int idx1 = line.indexOf(',');
    int idx2 = line.indexOf(',', idx1 + 1);

    if (idx1 <= 0 || idx2 <= idx1) continue;

    int hip   = line.substring(0, idx1).toInt();
    int knee  = line.substring(idx1 + 1, idx2).toInt();
    int ankle = line.substring(idx2 + 1).toInt();

    hip   = constrain(hip, HIP_MIN, HIP_MAX);
    knee  = constrain(knee, KNEE_MIN, KNEE_MAX);
    ankle = constrain(ankle, ANKLE_MIN, ANKLE_MAX);

    Serial.print("Moving to Hip:");
    Serial.print(hip);
    Serial.print(" Knee:");
    Serial.print(knee);
    Serial.print(" Ankle:");
    Serial.print(ankle);
    Serial.print("...");

    hipL.write(hip);
    kneeL.write(knee);
    ankleL.write(ankle);

    delay(2000);
    Serial.println(" movement is done");
    delay(3000);
    count++;
  }

  f.close();
  Serial.print("Done. Executed ");
  Serial.print(count);
  Serial.println(" position(s).");
}

void loop() {
  // Run once from setup; loop is idle
  delay(1000);
}
