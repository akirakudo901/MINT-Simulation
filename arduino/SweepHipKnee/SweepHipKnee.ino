#include <Servo.h>

Servo hipL;
Servo hipR;
Servo kneeL;
Servo kneeR;
Servo ankleL;
Servo ankleR;

#define hipLOffset 105
#define kneeLOffset 155
#define ankleLOffset 85

void sweepHipKneeTogether(Servo &hip, Servo &knee, int intervalDeg) {
  // Hip: 180 down to 0 by -interval; Knee: 0 up to 180 by +interval
  int steps = 180 / intervalDeg;

  for (int i = 0; i <= steps; i++) {
    int hipAngle = 180 - (i * intervalDeg);
    int kneeAngle = i * intervalDeg;

    Serial.print("Hip: ");
    Serial.print(hipAngle);
    Serial.print(" deg, Knee: ");
    Serial.print(kneeAngle);
    Serial.print(" deg...");

    hip.write(hipAngle);
    knee.write(kneeAngle);
    delay(1000);

    Serial.println("DONE.");
    delay(3000);
  }
}

void setup() {
  Serial.begin(9600);
  hipL.attach(10);
  kneeL.attach(9);
  ankleL.attach(8);

  // hipL.write(hipLOffset);
  // kneeL.write(kneeLOffset);
  // ankleL.write(ankleLOffset);
  hipL.write(0);
  delay(2000);
  kneeL.write(0);
  ankleL.write(0);
  delay(5000);
}

void loop() {
  // sweepHipKneeTogether(hipL, kneeL, 15);
  // delay(2000);

  // sweepHipKneeTogether(hipL, kneeL, -15);
  // delay(2000);

  hipL.write(0);
  kneeL.write(150);
  delay(5000);

  hipL.write(180);
  kneeL.write(150);
  delay(5000);
}
