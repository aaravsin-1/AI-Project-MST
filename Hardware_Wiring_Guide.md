# 🔌 COMPLETE HARDWARE WIRING GUIDE
## REES52 DS18B20 Temperature Sensor + REES52 L9110 Fan Module

---

# 📦 HARDWARE COMPONENTS

## 1. REES52 DS18B20 Temperature Sensor Module

### Specifications:
- **Sensor Type**: Digital temperature sensor (OneWire protocol)
- **Temperature Range**: -55°C to +125°C
- **Accuracy**: ±0.5°C (-10°C to +85°C range)
- **Resolution**: 9-bit to 12-bit selectable (0.0625°C at 12-bit)
- **Conversion Time**: 750ms (at 12-bit resolution)
- **Interface**: OneWire (single data line + power)
- **Operating Voltage**: 3.0V to 5.5V
- **Pull-up Resistor**: 4.7kΩ required (often built into module)

### Module Features:
- Pre-mounted DS18B20 sensor
- Built-in pull-up resistor (verify on your module!)
- 3 wires: VCC, GND, DATA
- Waterproof versions available
- Long cable options (1m, 2m, 3m)

### Why DS18B20 vs DHT11:

| Feature | DHT11 (Old) | DS18B20 (New) |
|---------|-------------|---------------|
| **Accuracy** | ±2°C | ±0.5°C |
| **Resolution** | 1°C | 0.0625°C (12-bit) |
| **Range** | 0-50°C | -55 to +125°C |
| **Interface** | Custom protocol | OneWire (standard) |
| **Reliability** | Moderate | High |
| **ML Benefit** | Basic | Better training data! |

**ML Impact**: DS18B20's higher precision (16x better!) provides cleaner training data!

---

## 2. REES52 L9110 Fan Module

### Specifications:
- **Chip**: L9110S dual H-bridge motor driver
- **Operating Voltage**: 2.5V to 12V
- **Current per Channel**: Up to 800mA continuous
- **Peak Current**: 1.5A (brief)
- **Control**: 2 pins per motor (direction + PWM speed)
- **Channels**: 2 (can control 2 motors independently)
- **Logic Level**: 5V compatible
- **Thermal Protection**: Yes (automatic shutdown at high temp)

### Module Pinout:
```
L9110 Module Pins:
┌────────────────┐
│ VCC  GND       │  Power input (5-12V)
│ A-IA A-IB      │  Motor A control
│ B-IA B-IB      │  Motor B control
│ MOTOR A+ A-    │  Motor A output
│ MOTOR B+ B-    │  Motor B output
└────────────────┘
```

### Why L9110 vs Direct PWM:

| Feature | Direct PWM | L9110 Module |
|---------|------------|--------------|
| **Current** | ~20mA | 800mA |
| **Protection** | None | Thermal + overcurrent |
| **Direction** | One-way | Bidirectional |
| **Safety** | Risk to Arduino | Isolated |
| **Power** | Weak fans only | Strong fans OK |

**Safety**: L9110 protects Arduino from motor back-EMF and overcurrent!

---

# 🔌 COMPLETE WIRING DIAGRAM

## Full System Connection:

```
┌─────────────────────────────────────────────────────────────────┐
│                         ARDUINO UNO                              │
│                                                                   │
│  ┌────────────────────────────────────────────────────┐          │
│  │                                                    │          │
│  │  5V    ──────┬──────────────┬────────────────────┐│          │
│  │  GND   ──────┼──────┬───────┼────────────┬───────┐││         │
│  │  Pin 2 ──────┼──────┼───────┼────────────┼───────┼┼┼──┐      │
│  │  Pin 5 ──────┼──────┼───────┼────────────┼───────┼┼┼──┼──┐   │
│  │  Pin 6 ──────┼──────┼───────┼────────────┼───────┼┼┼──┼──┼─┐ │
│  │              │      │       │            │       │││  │  │ │ │
│  └──────────────┼──────┼───────┼────────────┼───────┼┼┼──┼──┼─┼─┘
│                 │      │       │            │       │││  │  │ │
└─────────────────┼──────┼───────┼────────────┼───────┼┼┼──┼──┼─┼──
                  │      │       │            │       │││  │  │ │
                  │      │       │            │       │││  │  │ │
        ┌─────────▼──────▼───┐  │  ┌─────────▼───────▼┼┼──┼──┼─┘
        │  DS18B20 MODULE    │  │  │  L9110 FAN MODULE │││  │  │
        ├────────────────────┤  │  ├───────────────────┼┼┼──┼──┘
        │ VCC   GND   DATA   │  │  │ VCC  GND  A-IA A-IB│││  │
        │  │     │      │    │  │  │  │    │    │    │  │││  │
        │  │     │      └────┼──┘  │  │    │    │    │  │││  │
        │  │     │           │     │  │    │    └────┼──┘││  │
        │  │     └───────────┼─────┼──┘    └─────────┼───┘│  │
        │  └─────────────────┘     │                 └────┘  │
        │                          │                          │
        │  4.7kΩ Pull-up          │   MOTOR A+    A-         │
        │  (VCC to DATA)          │      │         │         │
        └──────────────────────────┘      │         │         │
                                          │         │         │
                                    ┌─────▼─────────▼──────┐  │
                                    │     COOLING FAN      │  │
                                    │   (DC Motor 5-12V)   │  │
                                    └──────────────────────┘  │
                                                               │
                         ┌─────────────────────────────────────┘
                         │
                    ┌────▼─────┐
                    │ Optional │
                    │ External │  For high-power fans
                    │  5-12V   │  (> 500mA)
                    │  Supply  │
                    └──────────┘
```

---

# 📝 STEP-BY-STEP WIRING INSTRUCTIONS

## Step 1: DS18B20 Temperature Sensor

### Identify the Wires:
- **Red**: VCC (Power +)
- **Black**: GND (Ground -)
- **Yellow/White**: DATA (OneWire signal)

### Connections:
```
DS18B20 → Arduino
─────────────────
Red     → 5V
Black   → GND
Yellow  → Pin 2
```

### CRITICAL: Pull-up Resistor
```
Check if your module has built-in resistor:
- Look for small resistor on module PCB
- Usually labeled "4.7K" or "R1"

If NO built-in resistor:
  Add 4.7kΩ resistor between:
    Yellow (DATA) ←─────[4.7kΩ]─────→ Red (VCC)
```

### Test DS18B20:
```cpp
// Upload test sketch:
#include <OneWire.h>
#include <DallasTemperature.h>

OneWire oneWire(2);
DallasTemperature sensors(&oneWire);

void setup() {
  Serial.begin(9600);
  sensors.begin();
}

void loop() {
  sensors.requestTemperatures();
  Serial.println(sensors.getTempCByIndex(0));
  delay(1000);
}

// Expected output: Room temperature (20-30°C)
```

---

## Step 2: L9110 Fan Module

### Identify the Pins:

**Control Side (to Arduino):**
```
┌─────────────┐
│ VCC  GND    │ ← Power for logic (5V from Arduino)
│ A-IA A-IB   │ ← Control signals
│ B-IA B-IB   │ ← (We only use Motor A)
└─────────────┘
```

**Motor Side (to Fan):**
```
┌─────────────┐
│ MOTOR A+ A- │ ← Fan connects here
│ MOTOR B+ B- │ ← (Unused)
└─────────────┘
```

### Connections:

**Power & Control:**
```
L9110   → Arduino
─────────────────
VCC     → 5V*
GND     → GND
A-IA    → Pin 5 (PWM speed control)
A-IB    → Pin 6 (direction control)
B-IA    → (not connected)
B-IB    → (not connected)

*For low-power fans (<500mA)
```

**Motor Output:**
```
L9110     → Fan
──────────────────
MOTOR A+  → Fan + (red wire)
MOTOR A-  → Fan - (black wire)
```

### Test L9110:
```cpp
// Upload test sketch:
#define FAN_IA 5
#define FAN_IB 6

void setup() {
  pinMode(FAN_IA, OUTPUT);
  pinMode(FAN_IB, OUTPUT);
}

void loop() {
  // Full speed
  analogWrite(FAN_IA, 255);
  digitalWrite(FAN_IB, LOW);
  delay(3000);
  
  // Half speed
  analogWrite(FAN_IA, 128);
  digitalWrite(FAN_IB, LOW);
  delay(3000);
  
  // Off
  analogWrite(FAN_IA, 0);
  digitalWrite(FAN_IB, LOW);
  delay(3000);
}

// Expected: Fan runs at different speeds
```

---

## Step 3: Optional External Power

### When to Use External Power:

**Use Arduino 5V if:**
- Fan current < 500mA
- Single small fan
- Short runtime

**Use External Power if:**
- Fan current > 500mA
- Large/powerful fan
- Continuous operation
- Multiple fans

### External Power Wiring:

```
External Supply → L9110
────────────────────────
+5V to +12V    → VCC (on L9110)
GND            → GND (on L9110)
                 │
                 └──→ GND (on Arduino) ← CRITICAL!

Important: Common ground between all components!
```

### Power Supply Selection:

```
Fan Current → Recommended Supply
─────────────────────────────────
< 500mA     → Arduino USB (500mA limit)
500-800mA   → External 5V 1A adapter
> 800mA     → External 12V 1-2A adapter
             + Use 12V fan
```

---

# ✅ VERIFICATION CHECKLIST

## Before Powering On:

- [ ] DS18B20 red wire to Arduino 5V
- [ ] DS18B20 black wire to Arduino GND
- [ ] DS18B20 yellow wire to Arduino Pin 2
- [ ] 4.7kΩ pull-up resistor present (built-in or external)
- [ ] L9110 VCC to Arduino 5V (or external supply)
- [ ] L9110 GND to Arduino GND
- [ ] L9110 A-IA to Arduino Pin 5
- [ ] L9110 A-IB to Arduino Pin 6
- [ ] Fan + to L9110 MOTOR A+
- [ ] Fan - to L9110 MOTOR A-
- [ ] If using external power: GND connected to Arduino GND
- [ ] No short circuits (check with multimeter)
- [ ] Correct polarity (+ to +, - to -)

## After Powering On:

- [ ] Arduino LED lights up
- [ ] No smoke or burning smell
- [ ] No excessive heat from components
- [ ] DS18B20 test code reads temperature
- [ ] L9110 test code runs fan
- [ ] Fan speed changes with PWM value
- [ ] Production firmware uploads successfully
- [ ] Python script detects Arduino
- [ ] Temperature readings make sense (15-35°C room temp)
- [ ] Fan responds to Python commands

---

# 🔧 TROUBLESHOOTING GUIDE

## DS18B20 Issues:

### Problem: "No DS18B20 sensor detected"
**Solutions:**
1. Check wiring (especially DATA to Pin 2)
2. Verify 4.7kΩ pull-up resistor
3. Test with different DS18B20 module
4. Check power supply (5V present?)
5. Try different OneWire pin (change in code)

### Problem: Temperature reads -127°C or NaN
**Solutions:**
1. Sensor disconnected - check wiring
2. Bad sensor - try replacement
3. Insufficient power - check 5V supply
4. Pull-up resistor missing

### Problem: Temperature fluctuates wildly
**Solutions:**
1. Add 0.1µF capacitor across VCC-GND
2. Keep wires short (<1m for DATA)
3. Check for electrical noise sources
4. Use shielded cable if necessary

---

## L9110 Issues:

### Problem: Fan doesn't spin
**Solutions:**
1. Check L9110 power (VCC needs 5-12V)
2. Verify fan polarity (swap A+ and A-)
3. Increase PWM value (try 255 first)
4. Check if fan needs minimum voltage to start
5. Test fan directly with power supply
6. Check Pin 5 and Pin 6 connections

### Problem: Fan runs at full speed always
**Solutions:**
1. Check A-IB is LOW (not floating)
2. Verify PWM on A-IA (not constant HIGH)
3. Test with different PWM values in code
4. Check Arduino pin isn't damaged

### Problem: Fan stutters or makes noise
**Solutions:**
1. Add 100µF capacitor across motor terminals
2. Enable rate limiting in software (±20/sec)
3. Check power supply can provide enough current
4. Add 0.1µF capacitor across VCC-GND on L9110

### Problem: L9110 gets hot
**Solutions:**
1. Normal if drawing high current
2. Add heatsink if very hot (>60°C)
3. Ensure proper ventilation
4. Check fan isn't stalled/blocked
5. Reduce duty cycle if possible

---

## Python/Arduino Communication Issues:

### Problem: "Arduino not available"
**Solutions:**
1. Check USB cable connection
2. Verify correct COM port / /dev/ttyUSB0
3. Try different ports in code
4. Check Arduino appears in device manager
5. Install CH340/FTDI drivers if needed

### Problem: "DS18B20 timeout"
**Solutions:**
1. Increase timeout in code (1.0s → 1.5s)
2. Check sensor is responding
3. Verify conversion time setting (12-bit = 750ms)

### Problem: Fan doesn't respond to commands
**Solutions:**
1. Check serial baud rate (9600)
2. Verify Arduino firmware uploaded
3. Test with Arduino Serial Monitor first
4. Check for serial buffer overflow

---

# 📊 EXPECTED PERFORMANCE

## DS18B20 Temperature Readings:

```
Resolution: 12-bit
Precision:  0.0625°C steps
Accuracy:   ±0.5°C

Example readings:
  24.0000°C
  24.0625°C  ← Notice precision!
  24.1250°C
  24.1875°C
```

## L9110 Fan Control:

```
PWM Value  Fan Speed  Voltage (12V supply)
─────────────────────────────────────────
0          OFF        0V
64         25%        3V
128        50%        6V
192        75%        9V
255        100%       12V

Rate Limiting:
  Max change: ±20 per second
  255 → 0 takes: ~13 seconds (smooth!)
```

---

# 🎯 FINAL ASSEMBLY TIPS

1. **Use breadboard first** - Test before soldering
2. **Label wires** - Use colored tape or labels
3. **Secure connections** - Heat shrink or electrical tape
4. **Cable management** - Zip ties for neatness
5. **Test incrementally** - One component at a time
6. **Document changes** - Take photos of working setup
7. **Keep spares** - Extra sensors and modules
8. **Protect from shorts** - Use electrical tape on exposed connections

---

# 📚 LIBRARY INSTALLATION

## Arduino Libraries Needed:

```cpp
// In Arduino IDE:
// Tools → Manage Libraries → Search and install:

1. OneWire by Paul Stoffregen
   - For DS18B20 communication
   
2. DallasTemperature by Miles Burton
   - For DS18B20 temperature reading
```

## Python Libraries Needed:

```bash
pip install pyserial psutil numpy pandas joblib
```

---

# ✨ HARDWARE UPGRADE BENEFITS

## Compared to DHT11 + Direct PWM:

| Metric | Old (DHT11 + PWM) | New (DS18B20 + L9110) |
|--------|-------------------|----------------------|
| **Temp Accuracy** | ±2°C | ±0.5°C (4x better!) |
| **Temp Resolution** | 1°C | 0.0625°C (16x better!) |
| **ML Training Data** | Noisy | Clean, precise |
| **Model RMSE** | ~1.5°C | ~1.0°C (estimate) |
| **Fan Control** | Direct (risky) | Protected, safe |
| **Max Fan Current** | 20mA | 800mA (40x more!) |
| **System Reliability** | Moderate | High |

**Bottom Line: Better sensors → Better data → Better ML model! 🎯**

---

**Wiring complete! Ready for production deployment!** 🚀