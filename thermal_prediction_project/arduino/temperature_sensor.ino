/*
 * DS18B20 Temperature Sensor & L9110 Fan Control - PRODUCTION VERSION
 * ===================================================================
 * Arduino firmware with safety fallback mechanism.
 * 
 * Hardware:
 * - REES52 DS18B20 Temperature Sensor Module (OneWire, Pin 2)
 * - REES52 L9110 Fan Module (Pins 5 & 6)
 * - Serial USB communication (9600 baud)
 * 
 * DS18B20 Specifications:
 * - Temperature range: -55°C to +125°C
 * - Accuracy: ±0.5°C (-10°C to +85°C)
 * - Resolution: 9-12 bit (0.0625°C at 12-bit)
 * - Interface: OneWire (single data line)
 * 
 * L9110 Fan Module Specifications:
 * - Dual H-bridge motor driver
 * - Operating voltage: 2.5V-12V
 * - Control: 2 pins (A-IA, A-IB) for direction and speed
 * - PWM for speed control
 * 
 * CRITICAL FIX: Safety fallback - if no command for 5 seconds,
 * automatically sets fan to safe speed (50%).
 */

#include <OneWire.h>
#include <DallasTemperature.h>

// ============================================================================
// PIN DEFINITIONS
// ============================================================================
#define ONE_WIRE_BUS 2    // DS18B20 data pin (OneWire)
#define FAN_IA 5          // L9110 Motor A - Input A (PWM speed control)
#define FAN_IB 6          // L9110 Motor A - Input B (direction/brake)

// ============================================================================
// SAFETY SETTINGS
// ============================================================================
#define TIMEOUT_MS 5000         // 5 seconds without command
#define SAFE_FAN_SPEED 128      // 50% speed as safe default

// ============================================================================
// DS18B20 SETUP
// ============================================================================
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// Device address (will be detected automatically)
DeviceAddress tempSensor;

// ============================================================================
// STATE VARIABLES
// ============================================================================
unsigned long lastCommandTime = 0;
int currentFanSpeed = 0;
bool safetyModeActive = false;

void setup() {
    // Initialize serial communication
    Serial.begin(9600);
    
    // ========================================================================
    // INITIALIZE DS18B20 TEMPERATURE SENSOR
    // ========================================================================
    sensors.begin();
    
    // Detect sensor
    int deviceCount = sensors.getDeviceCount();
    Serial.print("DS18B20 devices found: ");
    Serial.println(deviceCount);
    
    if (deviceCount == 0) {
        Serial.println("ERROR: No DS18B20 sensor detected!");
        Serial.println("Check wiring:");
        Serial.println("  - Red wire (VCC) → 5V");
        Serial.println("  - Black wire (GND) → GND");
        Serial.println("  - Yellow wire (DATA) → Pin 2");
        Serial.println("  - 4.7kΩ resistor between VCC and DATA");
    } else {
        // Get address of first sensor
        if (sensors.getAddress(tempSensor, 0)) {
            Serial.print("Sensor address: ");
            for (uint8_t i = 0; i < 8; i++) {
                if (tempSensor[i] < 16) Serial.print("0");
                Serial.print(tempSensor[i], HEX);
            }
            Serial.println();
            
            // Set resolution to 12-bit (0.0625°C precision)
            sensors.setResolution(tempSensor, 12);
            Serial.println("Resolution: 12-bit (0.0625°C)");
        }
    }
    
    // ========================================================================
    // INITIALIZE L9110 FAN MODULE
    // ========================================================================
    pinMode(FAN_IA, OUTPUT);
    pinMode(FAN_IB, OUTPUT);
    
    // Set initial fan speed to safe default
    setFanSpeed(SAFE_FAN_SPEED);
    currentFanSpeed = SAFE_FAN_SPEED;
    
    // Record initialization time
    lastCommandTime = millis();
    
    // Startup message
    Serial.println();
    Serial.println("========================================");
    Serial.println("Arduino Thermal Control v3.0 - PRODUCTION");
    Serial.println("Hardware: DS18B20 + L9110");
    Serial.println("Safety fallback: ENABLED");
    Serial.println("========================================");
}

void loop() {
    // Check for incoming serial commands
    if (Serial.available() > 0) {
        char command = Serial.read();
        
        // Update last command time
        lastCommandTime = millis();
        
        // Exit safety mode if active
        if (safetyModeActive) {
            safetyModeActive = false;
            Serial.println("Safety mode: OFF");
        }
        
        // Process command
        if (command == 'T') {
            // COMMAND: Get Temperature
            handleTemperatureRequest();
        }
        else if (command == 'F') {
            // COMMAND: Set Fan speed
            handleFanControl();
        }
        else if (command == 'S') {
            // COMMAND: Get Status
            handleStatusRequest();
        }
        else {
            // Unknown command
            Serial.println("ERROR: Unknown command");
        }
    }
    
    // ========================================================================
    // CRITICAL SAFETY FEATURE
    // ========================================================================
    // Check if Python has crashed or stopped sending commands
    unsigned long timeSinceLastCommand = millis() - lastCommandTime;
    
    if (timeSinceLastCommand > TIMEOUT_MS && !safetyModeActive) {
        // SAFETY FALLBACK TRIGGERED!
        safetyModeActive = true;
        currentFanSpeed = SAFE_FAN_SPEED;
        setFanSpeed(SAFE_FAN_SPEED);
        
        Serial.println();
        Serial.println("⚠⚠⚠ SAFETY MODE ACTIVATED! ⚠⚠⚠");
        Serial.print("Time since last command: ");
        Serial.print(timeSinceLastCommand / 1000);
        Serial.println(" seconds");
        Serial.print("Fan set to safe speed: ");
        Serial.print(SAFE_FAN_SPEED);
        Serial.println("/255 (50%)");
        Serial.println("Waiting for Python to reconnect...");
    }
    
    // Small delay to prevent serial buffer overflow
    delay(10);
}

// ============================================================================
// TEMPERATURE READING (DS18B20)
// ============================================================================
void handleTemperatureRequest() {
    /*
     * Read DS18B20 sensor and send temperature to Python.
     * 
     * DS18B20 Features:
     * - High precision: ±0.5°C accuracy
     * - 12-bit resolution: 0.0625°C steps
     * - Wide range: -55°C to +125°C
     * 
     * Response format: "24.5625\n" (temperature in Celsius)
     */
    
    // Request temperature reading
    sensors.requestTemperatures();
    
    // Get temperature in Celsius
    float temperature = sensors.getTempC(tempSensor);
    
    // Check if reading is valid
    // DS18B20 returns -127.00 on error
    if (temperature == DEVICE_DISCONNECTED_C || temperature < -55.0 || temperature > 125.0) {
        Serial.println("ERROR");
    } else {
        // Send temperature (DS18B20 provides high precision)
        Serial.println(temperature, 4);  // 4 decimal places
    }
}

// ============================================================================
// FAN CONTROL (L9110 H-Bridge Module)
// ============================================================================
void handleFanControl() {
    /*
     * Set fan speed using L9110 dual H-bridge module.
     * 
     * L9110 Control Logic:
     * ---------------------
     * For forward rotation (cooling):
     * - IA (Pin 5): PWM signal (speed control)
     * - IB (Pin 6): LOW (direction)
     * 
     * Speed Control:
     * - 0   = Fan OFF
     * - 128 = 50% speed
     * - 255 = 100% speed (maximum)
     * 
     * Command format: 'F' followed by speed (0-255)
     * Example: "F192" sets fan to 75% speed
     */
    
    // Read speed value from serial
    int speed = Serial.parseInt();
    
    // Validate speed range
    if (speed >= 0 && speed <= 255) {
        // Set fan speed using L9110
        setFanSpeed(speed);
        currentFanSpeed = speed;
        
        // Confirmation
        Serial.print("OK: Fan set to ");
        Serial.print(speed);
        Serial.print("/255 (");
        Serial.print((speed * 100) / 255);
        Serial.println("%)");
    } else {
        // Invalid speed
        Serial.print("ERROR: Invalid speed ");
        Serial.print(speed);
        Serial.println(" (must be 0-255)");
    }
}

// ============================================================================
// STATUS REQUEST
// ============================================================================
void handleStatusRequest() {
    /*
     * Send current system status to Python.
     * Useful for debugging and monitoring.
     */
    Serial.println("========================================");
    Serial.println("SYSTEM STATUS");
    Serial.println("========================================");
    
    // Sensor status
    Serial.print("DS18B20 Sensor: ");
    sensors.requestTemperatures();
    float temp = sensors.getTempC(tempSensor);
    if (temp == DEVICE_DISCONNECTED_C) {
        Serial.println("DISCONNECTED");
    } else {
        Serial.print("OK (");
        Serial.print(temp, 2);
        Serial.println("°C)");
    }
    
    // Fan status
    Serial.print("L9110 Fan Speed: ");
    Serial.print(currentFanSpeed);
    Serial.print("/255 (");
    Serial.print((currentFanSpeed * 100) / 255);
    Serial.println("%)");
    
    // Safety mode
    Serial.print("Safety Mode: ");
    Serial.println(safetyModeActive ? "ACTIVE" : "INACTIVE");
    
    // Uptime
    Serial.print("Uptime: ");
    Serial.print(millis() / 1000);
    Serial.println(" seconds");
    
    // Last command
    Serial.print("Time since last command: ");
    Serial.print((millis() - lastCommandTime) / 1000);
    Serial.println(" seconds");
    
    Serial.println("========================================");
}

// ============================================================================
// L9110 FAN SPEED CONTROL FUNCTION
// ============================================================================
void setFanSpeed(int speed) {
    /*
     * Control L9110 motor driver for fan speed.
     * 
     * L9110 H-Bridge Truth Table:
     * ---------------------------
     * IA  | IB  | Motor State
     * ----|-----|-------------
     * LOW | LOW | Motor OFF (brake)
     * PWM | LOW | Forward (cooling) - OUR MODE
     * LOW | PWM | Reverse (not used)
     * PWM | PWM | Motor OFF
     * 
     * For fan cooling, we use:
     * - IA (Pin 5) = PWM (0-255) for speed
     * - IB (Pin 6) = LOW for forward direction
     */
    
    if (speed == 0) {
        // Fan OFF - both pins LOW (brake mode)
        analogWrite(FAN_IA, 0);
        digitalWrite(FAN_IB, LOW);
    } else {
        // Fan ON - PWM on IA, LOW on IB (forward mode)
        analogWrite(FAN_IA, speed);
        digitalWrite(FAN_IB, LOW);
    }
}

/*
 * ============================================================================
 * HARDWARE WIRING GUIDE
 * ============================================================================
 * 
 * DS18B20 TEMPERATURE SENSOR MODULE:
 * -----------------------------------
 * DS18B20 Pin  →  Arduino Pin
 * VCC (Red)    →  5V
 * GND (Black)  →  GND
 * DATA (Yellow)→  Pin 2
 * 
 * IMPORTANT: Add 4.7kΩ pull-up resistor between VCC and DATA
 * (Some modules have built-in resistor - check yours!)
 * 
 * L9110 FAN MODULE:
 * -----------------
 * L9110 Pin    →  Arduino Pin
 * VCC          →  5V (or external 5-12V for more power)
 * GND          →  GND
 * A-IA         →  Pin 5 (PWM - speed control)
 * A-IB         →  Pin 6 (direction control)
 * MOTOR A+     →  Fan positive (+)
 * MOTOR A-     →  Fan negative (-)
 * 
 * NOTES:
 * - For higher power fans (>500mA), use external power supply
 * - Connect external GND to Arduino GND (common ground)
 * - L9110 can drive up to 800mA per channel
 * 
 * POWER OPTIONS:
 * --------------
 * Option 1: Low power fan (<500mA)
 *   - Power L9110 VCC from Arduino 5V
 *   - Simple, no external supply needed
 * 
 * Option 2: High power fan (>500mA)
 *   - Power L9110 VCC from external 5-12V supply
 *   - Connect external GND to Arduino GND
 *   - Arduino only provides control signals
 * 
 * ============================================================================
 * SAFETY MECHANISM EXPLANATION
 * ============================================================================
 * 
 * Problem: If Python script crashes, fan stays at last speed.
 *          Could be 0% when CPU is under heavy load → overheating!
 * 
 * Solution: Arduino monitors time since last command.
 *           If > 5 seconds, automatically sets fan to 50% (safe default).
 * 
 * Example Scenario:
 * -----------------
 * t=0s:   Python running, sending commands every 1s
 * t=10s:  Python crashes (exception, user kill, etc.)
 * t=15s:  Arduino notices: no command for 5s
 * t=15s:  Arduino activates safety mode: fan → 50%
 * Result: CPU stays cool even though Python crashed!
 * 
 * Why 50% (128/255)?
 * -------------------
 * - Not too low: Provides active cooling
 * - Not too high: Doesn't drain power
 * - Safe middle ground for most situations
 * - Adjustable based on your requirements
 * 
 * Recovery:
 * ---------
 * When Python restarts and sends command, Arduino automatically
 * exits safety mode and resumes normal operation.
 * 
 * ============================================================================
 * TROUBLESHOOTING
 * ============================================================================
 * 
 * DS18B20 Returns ERROR:
 * ----------------------
 * 1. Check wiring (especially DATA to Pin 2)
 * 2. Verify 4.7kΩ pull-up resistor is present
 * 3. Try different DS18B20 module (could be faulty)
 * 4. Check serial monitor for "devices found: 0"
 * 
 * Fan Doesn't Spin:
 * -----------------
 * 1. Check L9110 power supply (needs 5-12V)
 * 2. Verify fan is connected to MOTOR A terminals
 * 3. Try sending 'F255' for full speed test
 * 4. Check if fan needs minimum voltage to start
 * 5. Verify Pin 5 and Pin 6 connections
 * 
 * Safety Mode Activates Immediately:
 * ----------------------------------
 * 1. Python not sending commands
 * 2. Check serial port connection
 * 3. Verify baud rate is 9600
 * 4. Python script may have crashed
 * 
 * Temperature Reads -127°C:
 * -------------------------
 * 1. DS18B20 disconnected or faulty
 * 2. Check OneWire bus wiring
 * 3. Power supply issue
 * 
 * ============================================================================
 */
