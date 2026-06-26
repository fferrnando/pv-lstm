#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <BH1750.h>
#include <WiFi.h>
#include <PubSubClient.h>

// ============ KONFIGURASI ============
#define WIFI_SSID     "Redmi Note 12"
#define WIFI_PASSWORD "12345678910"
#define TB_HOST       "thingsboard.cloud"
#define TB_PORT       1883
#define TB_TOKEN      "wKjjXpRm8RoAK3ElKMwY"

// ============ PIN ============
#define ONE_WIRE_BUS 23   // DS18B20 di D23

// ============ INTERVAL ============
const unsigned long SEND_INTERVAL = 15000; // 15 detik → ThingsBoard
const unsigned long LCD_INTERVAL  = 2000;  // 2 detik  → refresh LCD

// ============ OBJEK ============
LiquidCrystal_I2C lcd(0x27, 16, 2);
Adafruit_INA219   ina219(0x40);
BH1750            lightMeter;
OneWire           oneWire(ONE_WIRE_BUS);
DallasTemperature sensorSuhu(&oneWire);
WiFiClient        espClient;
PubSubClient      client(TB_HOST, TB_PORT, espClient);

// ============ VARIABEL ============
unsigned long lastSend      = 0;
unsigned long lastLCD       = 0;
unsigned long lastReconnect = 0;
int lcdPage = 0;

// ============ FUNGSI WIFI ============
void initWiFi() {
  Serial.print("Connecting WiFi: ");
  Serial.println(WIFI_SSID);
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Connecting WiFi ");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
    if (millis() - start > 15000) {
      Serial.println("\nWiFi GAGAL!");
      lcd.setCursor(0, 1); lcd.print("WiFi GAGAL!     ");
      return;
    }
  }
  Serial.println("\nWiFi OK! IP: " + WiFi.localIP().toString());
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("WiFi OK!        ");
  lcd.setCursor(0, 1); lcd.print(WiFi.localIP().toString());
  delay(1500);
}

// ============ FUNGSI MQTT ============
void mqttConnect() {
  int retry = 0;
  while (!client.connected() && retry < 5) {
    retry++;
    Serial.print("Connecting MQTT... attempt ");
    Serial.println(retry);
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("Connect MQTT... ");
    lcd.setCursor(0, 1); lcd.print("Attempt: " + String(retry) + "       ");
    String clientId = "ESP32PLTS_" + String(random(0xffff), HEX);
    if (client.connect(clientId.c_str(), TB_TOKEN, NULL)) {
      Serial.println("MQTT OK!");
      lcd.clear();
      lcd.setCursor(0, 0); lcd.print("MQTT OK!        ");
      delay(800);
      lcd.clear();
    } else {
      Serial.println("GAGAL rc=" + String(client.state()));
      delay(3000);
    }
  }
}

// ============ FUNGSI KIRIM DATA ============
void sendTelemetry(float voltage, float current,
                   float power,   float temperature,
                   float lux) {
  if (!client.connected()) return;

  if (current  < 0) current  = 0.0f;
  if (power    < 0) power    = 0.0f;
  if (voltage  < 0) voltage  = 0.0f;
  if (lux      < 0) lux      = 0.0f;

  String payload = "{";
  payload += "\"voltage\":"     + String(voltage,     2) + ",";
  payload += "\"current\":"     + String(current,     3) + ",";
  payload += "\"power\":"       + String(power,       2) + ",";
  payload += "\"temperature\":" + String(temperature, 1) + ",";
  payload += "\"light\":"       + String(lux,         1);
  payload += "}";

  if (client.publish("v1/devices/me/telemetry", payload.c_str())) {
    Serial.println("TB OK: " + payload);
  } else {
    Serial.println("GAGAL kirim! Reconnect...");
    client.disconnect();
  }
}

// ============ UPDATE LCD (3 halaman bergantian) ============
void updateLCD(float voltage, float current,
               float power,   float suhu, float lux) {
  char baris1[17];
  char baris2[17];

  switch (lcdPage) {
    case 0:
      snprintf(baris1, sizeof(baris1), "V:%-5.2fV        ", voltage);
      snprintf(baris2, sizeof(baris2), "I:%-6.3fA       ", current);
      break;
    case 1:
      snprintf(baris1, sizeof(baris1), "W:%-5.2fW        ", power);
      snprintf(baris2, sizeof(baris2), "T:%-4.1fC        ", suhu);
      break;
    case 2:
      snprintf(baris1, sizeof(baris1), "Lux:%-8.1f    ", lux);
      snprintf(baris2, sizeof(baris2), "MQTT: %-10s", client.connected() ? "OK" : "OFF");
      break;
  }

  lcd.setCursor(0, 0); lcd.print(baris1);
  lcd.setCursor(0, 1); lcd.print(baris2);
  lcdPage = (lcdPage + 1) % 3;
}

// ============ SETUP ============
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== PLTS DATA LOGGER MULAI ===");
  Serial.println("Interval ThingsBoard : 15 detik");
  Serial.println("Interval LCD refresh : 2 detik");

  Wire.begin(21, 22);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("PLTS DataLogger ");
  lcd.setCursor(0, 1); lcd.print("Initializing... ");
  delay(1500);

  // INA219
  if (!ina219.begin()) {
    Serial.println("ERROR: INA219 tidak terbaca!");
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("ERROR: INA219   ");
    lcd.setCursor(0, 1); lcd.print("Cek kabel I2C!  ");
    delay(2000);
  } else {
    Serial.println("OK: INA219 0x40");
  }

  // ✅ BH1750 DENGAN MTREG 32 (OUTDOOR MODE)
  if (lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {

    // ✅ SET MTREG 32 UNTUK RANGE LEBIH TINGGI
    // Default MTreg = 69  → Batas ~54.000 lux (saturasi siang)
    // MTreg 32       → Batas ~120.000 lux (cocok outdoor!) ✅
    lightMeter.setMTreg(32);

    Serial.println("OK: BH1750 0x23");
    Serial.println("OK: MTreg=32 | Range ~120.000 lux | Outdoor Mode");
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("BH1750 OK!      ");
    lcd.setCursor(0, 1); lcd.print("Outdoor Mode    "); // ✅ Info MTreg
    delay(1000);
  } else {
    Serial.println("ERROR: BH1750 tidak terbaca!");
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("ERROR: BH1750   ");
    lcd.setCursor(0, 1); lcd.print("Cek ADDR-GND!   ");
    delay(2000);
  }

  // DS18B20
  sensorSuhu.begin();
  int jumlah = sensorSuhu.getDeviceCount();
  if (jumlah == 0) {
    Serial.println("ERROR: DS18B20 tidak terbaca!");
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("ERROR: DS18B20  ");
    lcd.setCursor(0, 1); lcd.print("Cek R 4.7k!     ");
    delay(2000);
  } else {
    Serial.println("OK: DS18B20 D23 (" + String(jumlah) + " sensor)");
  }

  client.setBufferSize(512);
  client.setSocketTimeout(10);

  randomSeed(esp_random());
  initWiFi();
  mqttConnect();

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Sistem Siap!    ");
  lcd.setCursor(0, 1); lcd.print("Interval: 15det ");
  delay(1500);
  lcd.clear();
}

// ============ LOOP ============
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi putus, reconnect...");
    initWiFi();
  }

  if (!client.connected()) {
    unsigned long nowR = millis();
    if (nowR - lastReconnect > 5000) {
      lastReconnect = nowR;
      mqttConnect();
    }
  }
  client.loop();

  unsigned long now = millis();

  // ===== BACA SEMUA SENSOR =====
  float busVoltage = ina219.getBusVoltage_V();
  float current_mA = ina219.getCurrent_mA();
  float current_A  = current_mA / 1000.0f;
  
  // HITUNG DAYA MURNI SECARA SOFTWARE (P = V x I)
  // Menghindari bug scaling pada register internal chip INA219
  float power_W    = busVoltage * current_A;

  delay(10);

  sensorSuhu.requestTemperatures();
  float suhu = sensorSuhu.getTempCByIndex(0);

  static float lux = 0.0f;
  if (lightMeter.measurementReady()) {
    float bacaan = lightMeter.readLightLevel();
    if (bacaan >= 0) lux = bacaan;
  }

  // ===== VALIDASI NILAI =====
  if (busVoltage < 0) busVoltage = 0.0f;
  if (current_A  < 0) current_A  = 0.0f;
  if (power_W    < 0) power_W    = 0.0f;
  if (lux        < 0) lux        = 0.0f;

  if (suhu == DEVICE_DISCONNECTED_C || suhu > 80.0 || suhu < -10.0) {
    Serial.println("WARNING: DS18B20 error!");
    suhu = 0.0f;
  }

  // ===== UPDATE LCD tiap 2 detik =====
  if (now - lastLCD >= LCD_INTERVAL) {
    lastLCD = now;
    updateLCD(busVoltage, current_A, power_W, suhu, lux);
  }

  // ===== KIRIM KE THINGSBOARD tiap 15 detik =====
  if (now - lastSend >= SEND_INTERVAL) {
    lastSend = now;
    Serial.printf(
      "V:%.2fV | I:%.3fA | W:%.2fW | T:%.1fC | Lux:%.1f\n",
      busVoltage, current_A, power_W, suhu, lux
    );
    sendTelemetry(busVoltage, current_A, power_W, suhu, lux);
  }
}