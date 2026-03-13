#include <Servo.h>

Servo hipL;
Servo hipR;
Servo kneeL;
Servo kneeR;
Servo ankleL;
Servo ankleR;

void setup() {
  Serial.begin(9600);
  hipL.attach(10);
  kneeL.attach(9);
  ankleL.attach(8);
  
  hipL.write(0); 
  // kneeL.write(0);
  // ankleL.write(0);
  
  delay(5000);
}

void loop() {
  hipL.write(90);
  delay(3000);
  
  hipL.write(45);
  delay(3000);

  hipL.write(0);
  delay(3000);
}