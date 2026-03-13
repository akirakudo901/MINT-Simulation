#include <Servo.h>

Servo hipL;
Servo hipR;
Servo kneeL;
Servo kneeR;
Servo ankleL;
Servo ankleR;

void sweepMotorRange(Servo &motor, int startDeg, int endDeg, int intervalDeg) {
  int step = (endDeg >= startDeg) ? intervalDeg : -intervalDeg;

  for (int angle = startDeg; (step > 0 && angle <= endDeg) || (step < 0 && angle >= endDeg); angle += step) {
    Serial.print("Moving to ");
    Serial.print(angle);
    Serial.print(" degrees...");

    motor.write(angle);
    delay(2000);

    Serial.println("DONE.");
    delay(500);
  }
}

void setup() {
  Serial.begin(9600);
  hipL.attach(10);
  kneeL.attach(9);
  ankleL.attach(8);

  hipL.write(0);
  delay(5000);
}

void loop() {
  // Example: sweep hipL from 0 to 90 degrees in 15-degree steps
  sweepMotorRange(hipL, 0, -180, 15);
  delay(2000);

  sweepMotorRange(kneeL, 0, -180, 15);
  delay(2000);

  sweepMotorRange(ankleL, 0, -180, 15);
  delay(2000);

  // // Example: sweep back from 90 to 0 in 15-degree steps
  // sweepMotorRange(hipL, 90, 0, 15);
  // delay(2000);
}
