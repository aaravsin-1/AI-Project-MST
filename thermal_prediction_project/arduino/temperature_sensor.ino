/*
 * Arduino DS18B20 Temperature Sensor Interface
 * ============================================
 * Reads ambient temperature from DS18B20 sensor
 * and sends data via Serial to Python data collector.
 * 
 * Hardware Connections:
 * - DS18B20 Data Pin -> Arduino Pin 2
 * - DS18B20 VCC -> Arduino 5V
 * - DS18B20 GND -> Arduino GND
 * - 4.7kΩ resistor between Data and VCC (pull-up)
 * 
 * Libraries Required:
 * - OneWire (by Paul Stoffregen)
 * - DallasTemperature (by Miles Burton)
 * 
 * Install via Arduino IDE: Tools > Manage Libraries > Search for above
 */

#include <OneWire.h>
#include <DallasTemperature.h>

// DS18B20 connected to pin 2
#define ONE_WIRE_BUS 2

// Setup OneWire instance
OneWire oneWire(ONE_WIRE_BUS);

// Pass OneWire reference to DallasTemperature library
DallasTemperature sensors(&oneWire);

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize DS18B20 sensor
  sensors.begin();
  
  // Wait for serial connection
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  Serial.println("Arduino Temperature Sensor Ready");
  Serial.println("Send 'T' to request temperature reading");
}

void loop() {
  // Check if data is available from Python
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    // If 'T' command received, read temperature
    if (command == 'T' || command == 't') {
      // Request temperature from sensor
      sensors.requestTemperatures();
      
      // Read temperature in Celsius
      float temperature = sensors.getTempCByIndex(0);
      
      // Send temperature to Python
      if (temperature != DEVICE_DISCONNECTED_C) {
        Serial.println(temperature, 2);  // Send with 2 decimal places
      } else {
        Serial.println("ERROR");
      }
    }
  }
  
  // Small delay to prevent overwhelming the serial buffer
  delay(10);
}

/*
 * ALTERNATIVE: Simple Version Without External Sensor
 * ===================================================
 * If you don't have a DS18B20 sensor, use this simplified version
 * that simulates ambient temperature.
 * 
 * Uncomment the code below and comment out everything above:
 */

/*
void setup() {
  Serial.begin(9600);
  randomSeed(analogRead(0));
  Serial.println("Arduino Simulated Temperature Sensor Ready");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    if (command == 'T' || command == 't') {
      // Simulate ambient temperature between 22-26°C
      float temperature = 24.0 + random(-20, 20) / 10.0;
      Serial.println(temperature, 2);
    }
  }
  
  delay(10);
}
*/

/*
 * OPTIONAL: Fan Control Addition
 * ==============================
 * To add predictive fan control, add these components:
 * 
 * Hardware:
 * - 5V/12V DC Fan
 * - NPN Transistor (e.g., 2N2222)
 * - 1kΩ resistor
 * - Flyback diode (1N4007)
 * 
 * Connections:
 * - Arduino Pin 9 -> 1kΩ resistor -> Transistor Base
 * - Transistor Collector -> Fan (-)
 * - Transistor Emitter -> GND
 * - Fan (+) -> External Power Supply (+)
 * - Diode across Fan (cathode to +, anode to -)
 * 
 * Add to setup():
 *   pinMode(9, OUTPUT);
 * 
 * Add new command in loop():
 *   if (command == 'F') {
 *     int speed = Serial.parseInt();  // Read fan speed (0-255)
 *     analogWrite(9, speed);
 *     Serial.println("OK");
 *   }
 */
