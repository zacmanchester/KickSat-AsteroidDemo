/*
  SerialDemo
  Demonstrates the use of the Serial port for sending information
  back to the host PC
*/

unsigned int tick = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  if(tick < 100)
  {
    Serial.println("3");
    ++tick;
  }
  else if(tick < 200)
  {
    Serial.println("-3");
    ++tick;
  }
  else
  {
    tick = 0;
  }
    
  delay(50);
}
