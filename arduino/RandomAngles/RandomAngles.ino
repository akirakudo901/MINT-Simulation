/*
 * RandomAngles - receives hip,knee,ankle triplets over Serial and applies to motors.
 * Expects format: hip,knee,ankle (one triplet per line, e.g. "45,90,60")
 * Use generate_random_angles.py to create angle file and send to Arduino.
 */

#include <Servo.h>

Servo hipL;
Servo hipR;
Servo kneeL;
Servo kneeR;
Servo ankleL;
Servo ankleR;

// Joint limits (clamp received values)
#define HIP_MIN 0
#define HIP_MAX 180
#define KNEE_MIN 0
#define KNEE_MAX 150
#define ANKLE_MIN 0
#define ANKLE_MAX 120

#define BAUDRATE 9600

#define HIP_RELAX 90
#define KNEE_RELAX 70
#define ANKLE_RELAX 90

void setup() {
  Serial.begin(BAUDRATE);
  hipL.attach(10);
  kneeL.attach(9);
  ankleL.attach(8);

  hipL.write(HIP_RELAX);
  kneeL.write(KNEE_RELAX);
  ankleL.write(ANKLE_RELAX);
  delay(1000);

  Serial.println("Ready. Send hip,knee,ankle (e.g. 45,90,60)");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    int hip = -1, knee = -1, ankle = -1;
    int idx1 = line.indexOf(',');
    int idx2 = line.indexOf(',', idx1 + 1);

    if (idx1 > 0 && idx2 > idx1) {
      hip = line.substring(0, idx1).toInt();
      knee = line.substring(idx1 + 1, idx2).toInt();
      ankle = line.substring(idx2 + 1).toInt();
    }

    if (hip >= 0 && knee >= 0 && ankle >= 0) {
      hip = constrain(hip, HIP_MIN, HIP_MAX);
      knee = constrain(knee, KNEE_MIN, KNEE_MAX);
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

      delay(1000);

      Serial.println(" movement is done");
      delay(1500);
    } else {
      Serial.println("Invalid format. Use: hip,knee,ankle");
    }
  }

  delay(1000);
  
  hipL.write(HIP_RELAX);
  kneeL.write(KNEE_RELAX);
  ankleL.write(ANKLE_RELAX);
}
