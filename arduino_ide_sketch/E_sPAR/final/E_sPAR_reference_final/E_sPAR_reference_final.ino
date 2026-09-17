// PARcast E_s(PAR) Reference Station
// File: E_sPAR_reference_final.ino
// Logs surface E_s(PAR) data to SD card with automatically incrementing filename
// Format: E_sPAR_YYYYMMDD_0001.CSV
// Columns: datetime, raw_counts, millivolts, ppfd_umol_m2_s

#include <Adafruit_ADS1X15.h>
#include <RTClib.h>
#include <SD.h>

Adafruit_ADS1115 ads;
RTC_DS3231 rtc;

char filename[30];

void generateFilename(DateTime now) {
  char dateStr[12];
  sprintf(dateStr, "%04d%02d%02d", now.year(), now.month(), now.day());
  
  int deployNum = 1;
  while (deployNum <= 9999) {
    sprintf(filename, "E_sPAR_%s_%04d.CSV", dateStr, deployNum);
    if (!SD.exists(filename)) {
      break;
    }
    deployNum++;
  }
}

void setup() {
  Serial.begin(9600);
  delay(3000);
  Serial.println("PARcast E_s(PAR) Reference Station Starting...");

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  if (!rtc.begin()) {
    Serial.println("ERROR: RTC failed! Check wiring.");
    while (1);
  }
  Serial.println("RTC initialized!");

  if (!ads.begin()) {
    Serial.println("ERROR: ADS1115 failed! Check wiring.");
    while (1);
  }
  ads.setGain(GAIN_SIXTEEN);
  Serial.println("ADS1115 initialized!");

  if (!SD.begin(BUILTIN_SDCARD)) {
    Serial.println("ERROR: SD card failed! Check card is inserted.");
    while (1);
  }
  Serial.println("SD card initialized!");

  DateTime now = rtc.now();
  generateFilename(now);

  File dataFile = SD.open(filename, FILE_WRITE);
  if (dataFile) {
    dataFile.println("datetime,raw_counts,millivolts,ppfd_umol_m2_s");
    dataFile.close();
    Serial.print("Logging to: ");
    Serial.println(filename);
  } else {
    Serial.println("ERROR: Could not create log file!");
    while (1);
  }

  Serial.println("------------------------------------------");
  Serial.println("datetime,raw_counts,millivolts,ppfd_umol_m2_s");
  Serial.println("------------------------------------------");
}

void loop() {
  DateTime now = rtc.now();
  char timestamp[26];
  sprintf(timestamp, "%04d-%02d-%02d %02d:%02d:%02d",
    now.year(), now.month(), now.day(),
    now.hour(), now.minute(), now.second());

  int16_t raw = ads.readADC_Differential_0_1();
  float millivolts = ads.computeVolts(raw) * 1000.0;
  float ppfd = millivolts * 100.0;

  Serial.print(timestamp);
  Serial.print(",");
  Serial.print(raw);
  Serial.print(",");
  Serial.print(millivolts, 4);
  Serial.print(",");
  Serial.println(ppfd, 2);

  File dataFile = SD.open(filename, FILE_WRITE);
  if (dataFile) {
    dataFile.print(timestamp);
    dataFile.print(",");
    dataFile.print(raw);
    dataFile.print(",");
    dataFile.print(millivolts, 4);
    dataFile.print(",");
    dataFile.println(ppfd, 2);
    dataFile.close();
  } else {
    Serial.println("ERROR: Could not open log file!");
  }

  digitalWrite(LED_BUILTIN, HIGH);
  delay(50);
  digitalWrite(LED_BUILTIN, LOW);
  delay(950);
}
