#include <SpriteMag.h>

SpriteMag mag = SpriteMag();

void setup() {
  mag.init();
  Serial.begin(9600);
}

void loop() {
  
  //Values are in Guass
  MagneticField b = mag.read();
  
  Serial.print(b.x);
  Serial.print("\t");
  Serial.print(b.y);
  Serial.print("\t");
  Serial.println(b.z);
  
  delay(190);
}
