#include <Servo.h>
#include "constants.h"

Servo hipL;
Servo hipR;
Servo kneeL;
Servo kneeR;
Servo ankleL;
Servo ankleR;

void updateServoPos(int target1, int target2, int target3, char leg){
  if (leg == 'l'){
    hipL.write(hipLOffset - target1);
    kneeL.write(kneeLOffset - target2);
    ankleL.write(2*ankleLOffset + target3 - 163);
  }
  else if (leg == 'r'){
    hipR.write(hipROffset + target1);
    kneeR.write(kneeROffset + target2);
    ankleR.write(2*ankleROffset + target3);
  }
}
void pos(float x, float z, char leg){
  float d = sqrt(x*x + z*z);

  // hip components
  float hipRad1 = acos((l1*l1 + d*d - l2*l2) / (2*l1*d));
  float hipRad2 = atan2(x, z);
  float hipRad = hipRad1 + hipRad2;
  float hipDeg = hipRad * 180.0 / PI;

  // knee
  float kneeRad = PI - acos((l1*l1 + l2*l2 - d*d) / (2*l1*l2));
  float kneeDeg = kneeRad * 180.0 / PI;

  // ankle (to keep foot level)
  float ankleRad = -(hipRad + kneeRad);
  float ankleDeg = ankleRad * 180.0 / PI;
if (leg == 'l') ankleDeg = -ankleDeg;
  updateServoPos(hipDeg, kneeDeg, ankleDeg, leg);
}


void takeStep(float stepLength, int stepVelocity){
  for(float t = 0; t < 360; t += 5){   // Smooth cyclic loop (no endpoints)
    float rad = t * PI / 180.0;

    float xr =  -stepLength * sin(rad);        // forward swing
    float xl = stepLength * sin(rad);        // opposite phase

    float zr = stepHeight + stepClearance * cos(rad);
    float zl = stepHeight + stepClearance * cos(rad + PI); // 180° shifted

    pos(xr, zr, 'r');
    pos(xl, zl, 'l');

    delay(20);
  }
}


void initialize(){
  for (float i = 10.7; i >= stepHeight; i-=0.1){
    pos(0, i, 'l');
    pos(0, i, 'r');
  }
}

void setup() {
  Serial.begin(9600);
  hipL.attach(5);
  kneeL.attach(6);
  ankleL.attach(7);
  hipR.attach(8);
  kneeR.attach(9);
  ankleR.attach(10);
 
  hipL.write(hipLOffset); 
  kneeL.write(kneeLOffset);
  ankleL.write(ankleLOffset);
  
  hipR.write(hipROffset);
  kneeR.write(kneeROffset);
  ankleR.write(ankleROffset);

  delay(5000);
  
  initialize();
}

void loop() {
  takeStep(2, 100);
}
