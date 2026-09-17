/*
 * E_dzPAR — PARcast underwater profiler data logger
 * Firmware version: 2026.08.05-v5
 * github.com/LoriAzniveBerberian/PARcast
 *
 * Logs downwelling PAR irradiance E_d(z, PAR) through the water column at
 * 8 Hz, with depth, orientation, and temperature, to a CSV file on microSD.
 *
 * SENSORS
 *   DS3231 RTC        real-time clock; supplies the wall-clock timestamp
 *   SQ-500 + ADS1115  PAR sensor (analog mV) read by a 16-bit ADC
 *   MS5837 (Bar30)    absolute pressure; gives depth and water temperature
 *   LSM6DSOX          accelerometer + gyroscope
 *   LIS3MDL           magnetometer
 *   Madgwick filter   fuses the two IMUs into roll / pitch / yaw
 *
 * OUTPUT
 *   /E_dzPAR_YYYYMMDD_NNNN.CSV, new file each boot, NNNN auto-incrementing
 *   per day. Matches the E_s(PAR) surface station naming so profiler and
 *   surface files pair by date and deploy number.
 *
 *   The file opens with six '#' metadata lines recording firmware version,
 *   sensor serial, calibration constant, ADC settings, and sample rate, so
 *   each data file identifies the configuration that produced it.
 *   Read with pandas.read_csv(path, comment='#').
 *
 *   Then 22 columns. Columns 1-20 match the earlier 20-column build;
 *   millis_boot and par_adc_counts are appended after them.
 *
 * BEFORE RUNNING
 *   1. Set the DS3231 with PARcast_RTC_set. Use the same timezone
 *      convention as the surface reference station.
 *   2. Run IMU_calibration and paste the magnetometer values below.
 *   3. Insert a FAT32-formatted microSD card.
 *   4. Only one .ino defining setup()/loop() may be in this folder.
 *
 * LIBRARIES
 *   Adafruit LSM6DSOX, LIS3MDL, ADS1X15, Unified Sensor, BusIO,
 *   RTClib, MS5837 (BlueRobotics), Madgwick, SD
 */

#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <Adafruit_LSM6DSOX.h>
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_Sensor.h>
#include <RTClib.h>
#include "MS5837.h"
#include <MadgwickAHRS.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// Written into the CSV metadata block. Bump on every edit to the sketch.
const char FIRMWARE_VERSION[] = "2026.08.05-v5";

// Rate at which rows are written to the CSV.
const float    SAMPLE_RATE_HZ     = 8.0;
const uint32_t SAMPLE_INTERVAL_MS = (uint32_t)(1000.0 / SAMPLE_RATE_HZ);

// Number of ADS1115 conversions averaged into each logged PAR value.
// Averaging recovers resolution below one ADC code. The ADC runs at 860 SPS
// (~1.2 ms per conversion) so all 16 reads fit inside the 125 ms sample
// interval alongside the Bar30 read and the SD write.
const uint8_t PAR_OVERSAMPLE = 16;

// Volts per ADC code at GAIN_SIXTEEN, expressed in mV. Converts raw counts
// to sensor voltage.
const float ADS_LSB_MV = 0.0078125;

// Sensor serial number, from the label near the pigtail leads.
const char SQ500_SERIAL[] = "6210";

// Converts sensor output in mV to PPFD in umol m-2 s-1.
// Fixed standard value for the SQ-500-SS (sensitivity 0.01 mV per
// umol m-2 s-1). Only the digital SQ-520/521/522 use per-sensor factors.
const float SQ500_CAL_FACTOR = 100.0;

// Immersion effect correction factor: scales underwater PPFD to absolute
// values, since more radiation backscatters out of the diffuser in water
// than in air. Apogee specifies 1.25 for serials 2876 and above, 1.32 for
// 0-2875. Recorded in the CSV metadata; applied in post-processing, not
// here, so the logged file stays the raw instrument record.
const float SQ500_IECF = 1.25;

// Water density used by the Bar30 to convert pressure to depth.
// 997 kg/m3 freshwater, 1029 kg/m3 seawater.
const float WATER_DENSITY = 1029.0;

// Hard-iron offsets and soft-iron scale factors for the magnetometer,
// from IMU_calibration.ino. Applied before the values reach the filter.
const float MAG_OFFSET_X = 0.0;
const float MAG_OFFSET_Y = 0.0;
const float MAG_OFFSET_Z = 0.0;
const float MAG_SCALE_X  = 1.0;
const float MAG_SCALE_Y  = 1.0;
const float MAG_SCALE_Z  = 1.0;

// Rate at which the Madgwick filter integrates IMU data. Runs faster than
// the sample rate so the orientation estimate has converged at each sample.
const float    FILTER_RATE_HZ     = 50.0;
const uint32_t FILTER_INTERVAL_US = (uint32_t)(1000000.0 / FILTER_RATE_HZ);

// Teensy 4.1 built-in microSD slot.
const int SD_CS_PIN = BUILTIN_SDCARD;

// ============================================================================

Adafruit_LSM6DSOX lsm6ds;
Adafruit_LIS3MDL  lis3mdl;
Adafruit_ADS1115  ads;
RTC_DS3231        rtc;
MS5837            bar30;
Madgwick          filter;
File              dataFile;

uint32_t lastSample       = 0;   // millis() of the last scheduled sample
uint32_t lastFilterUpdate = 0;   // micros() of the last filter integration
uint32_t sampleCount      = 0;   // rows written; drives flush and print intervals
uint32_t overrunCount     = 0;   // samples that exceeded the sample interval
char     filename[40];

// The DS3231 resolves only to whole seconds. millis() supplies the fraction:
// when the RTC second changes, millis() is re-captured, and the sub-second
// part of each timestamp is the elapsed millis() since that capture.
// Re-capturing every second stops millis() drift from accumulating.
uint32_t rtcAnchorUnix   = 0;
uint32_t rtcAnchorMillis = 0;

const int LED_PIN = LED_BUILTIN;

// Error codes are signalled by blink count, so a sealed instrument with no
// Serial connection still reports which subsystem failed.
void blinkError(int count) {
  for (int i = 0; i < count; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
  delay(1000);
}

void haltWithError(const char* msg, int blinkCount) {
  Serial.println(msg);
  while (1) {
    blinkError(blinkCount);
  }
}

// Fixed-width float printing, so the Serial preview stays column-aligned.
void printPadded(float value, int decimals, int width) {
  char buf[16];
  dtostrf(value, width, decimals, buf);
  Serial.print(buf);
  Serial.print(F("  "));
}

void setup() {
  pinMode(LED_PIN, OUTPUT);

  Serial.begin(115200);
  // Wait briefly for a USB host, then continue regardless so the instrument
  // starts logging when powered from a battery with no computer attached.
  while (!Serial && millis() < 3000) { ; }

  Serial.println();
  Serial.println(F("================================================"));
  Serial.print  (F("E_dzPAR — PARcast Data Logger   fw "));
  Serial.println(FIRMWARE_VERSION);
  Serial.println(F("================================================"));

  Wire.begin();
  Wire.setClock(400000);   // 400 kHz I2C: five devices are read every sample

  // ---- RTC: supplies the timestamp and the date used in the filename ----
  Serial.print(F("DS3231 RTC... "));
  if (!rtc.begin()) {
    haltWithError("FAILED", 2);
  }
  if (rtc.lostPower()) {
    Serial.println(F("WARNING — RTC lost power, time may be incorrect."));
  }
  DateTime now = rtc.now();
  Serial.print(F("OK ("));
  Serial.print(now.year()); Serial.print('-');
  if (now.month()  < 10) Serial.print('0'); Serial.print(now.month());  Serial.print('-');
  if (now.day()    < 10) Serial.print('0'); Serial.print(now.day());    Serial.print(' ');
  if (now.hour()   < 10) Serial.print('0'); Serial.print(now.hour());   Serial.print(':');
  if (now.minute() < 10) Serial.print('0'); Serial.print(now.minute()); Serial.print(':');
  if (now.second() < 10) Serial.print('0'); Serial.print(now.second());
  Serial.println(F(")"));

  // ---- ADS1115: digitizes the SQ-500's analog output ----
  Serial.print(F("ADS1115... "));
  if (!ads.begin()) {
    haltWithError("FAILED", 3);
  }
  // GAIN_SIXTEEN gives +/-0.256 V full scale. The SQ-500 outputs 0-40 mV,
  // so this is the most sensitive range that still covers the sensor.
  ads.setGain(GAIN_SIXTEEN);
  // 860 SPS is the fastest conversion rate, which is what makes
  // PAR_OVERSAMPLE reads per sample possible.
  ads.setDataRate(RATE_ADS1115_860SPS);
  Serial.print(F("OK (oversample x"));
  Serial.print(PAR_OVERSAMPLE);
  Serial.println(F(" @ 860 SPS)"));

  // ---- LSM6DSOX: accelerometer and gyroscope for the orientation filter ----
  Serial.print(F("LSM6DSOX... "));
  if (!lsm6ds.begin_I2C()) {
    haltWithError("FAILED", 4);
  }
  // 4 g and 500 dps cover hand-deployed motion without clipping.
  lsm6ds.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
  lsm6ds.setGyroRange(LSM6DS_GYRO_RANGE_500_DPS);
  // 104 Hz output keeps the IMU ahead of the 50 Hz filter rate.
  lsm6ds.setAccelDataRate(LSM6DS_RATE_104_HZ);
  lsm6ds.setGyroDataRate(LSM6DS_RATE_104_HZ);
  Serial.println(F("OK"));

  // ---- LIS3MDL: magnetometer, gives the filter an absolute heading ----
  Serial.print(F("LIS3MDL... "));
  if (!lis3mdl.begin_I2C()) {
    haltWithError("FAILED", 5);
  }
  lis3mdl.setPerformanceMode(LIS3MDL_HIGHMODE);
  lis3mdl.setOperationMode(LIS3MDL_CONTINUOUSMODE);
  lis3mdl.setDataRate(LIS3MDL_DATARATE_155_HZ);
  // 4 gauss is the most sensitive range; Earth's field is under 1 gauss.
  lis3mdl.setRange(LIS3MDL_RANGE_4_GAUSS);
  Serial.println(F("OK"));

  // ---- MS5837: pressure sensor providing depth and water temperature ----
  Serial.print(F("MS5837 Bar30... "));
  if (!bar30.init()) {
    haltWithError("FAILED", 6);
  }
  bar30.setModel(MS5837::MS5837_30BA);
  bar30.setFluidDensity(WATER_DENSITY);
  Serial.println(F("OK"));

  filter.begin(FILTER_RATE_HZ);
  Serial.println(F("Madgwick filter initialized"));

  // ---- microSD ----
  Serial.print(F("microSD... "));
  if (!SD.begin(SD_CS_PIN)) {
    haltWithError("FAILED — card missing or not FAT32?", 7);
  }
  Serial.println(F("OK"));

  // Find the lowest unused deploy number for today's date, so a new file is
  // created each boot and nothing already on the card is overwritten.
  {
    char dateStr[12];
    snprintf(dateStr, sizeof(dateStr), "%04d%02d%02d",
             now.year(), now.month(), now.day());
    int deployNum = 1;
    while (deployNum <= 9999) {
      snprintf(filename, sizeof(filename), "E_dzPAR_%s_%04d.CSV",
               dateStr, deployNum);
      if (!SD.exists(filename)) break;
      deployNum++;
    }
  }

  dataFile = SD.open(filename, FILE_WRITE);
  if (!dataFile) {
    haltWithError("FAILED to create log file", 8);
  }

  // Metadata block: records the configuration this file was logged under.
  dataFile.print(F("# instrument=PARcast_E_dzPAR firmware="));
  dataFile.println(FIRMWARE_VERSION);
  dataFile.print(F("# sq500_serial=")); dataFile.print(SQ500_SERIAL);
  dataFile.print(F(" sq500_cal_factor=")); dataFile.print(SQ500_CAL_FACTOR, 4);
  dataFile.println(F(" units=umol_m2_s_per_mV"));
  dataFile.print(F("# ads_gain=GAIN_SIXTEEN ads_lsb_mV=")); dataFile.print(ADS_LSB_MV, 7);
  dataFile.print(F(" ads_rate_sps=860 par_oversample=")); dataFile.println(PAR_OVERSAMPLE);
  dataFile.print(F("# sample_rate_hz=")); dataFile.print(SAMPLE_RATE_HZ, 2);
  dataFile.print(F(" water_density_kg_m3=")); dataFile.println(WATER_DENSITY, 1);
  dataFile.print(F("# immersion_correction=NOT_APPLIED apply_iecf_in_post="));
  dataFile.println(SQ500_IECF, 3);
  dataFile.println(F("# depth_note=subtract_surface_baseline_at_altitude"));

  dataFile.println(F(
    "iso_time,unix_time,par_mV,par_uMol_m2_s,depth_m,pressure_mbar,water_temp_C,"
    "accel_x_ms2,accel_y_ms2,accel_z_ms2,"
    "gyro_x_rads,gyro_y_rads,gyro_z_rads,"
    "mag_x_uT,mag_y_uT,mag_z_uT,"
    "roll_deg,pitch_deg,yaw_deg,"
    "imu_temp_C,millis_boot,par_adc_counts"
  ));
  dataFile.flush();

  Serial.print(F("Logging to: "));
  Serial.println(filename);
  Serial.println(F("CSV: 22 columns + metadata block."));
  Serial.println(F("PAR logged RAW — immersion factor applied in post-processing."));
  Serial.println();

  // The Madgwick filter converges over several seconds from its initial
  // guess, so it is run before logging starts to avoid a settling transient
  // in the first rows of orientation data.
  Serial.println(F("Warming up orientation filter (3s)..."));
  uint32_t warmupStart = millis();
  while (millis() - warmupStart < 3000) {
    updateFilter();
  }

  Serial.println(F("Logging started. LED blinks on every sample."));
  Serial.println();
  Serial.println(F(
    "iso_time                 PPFD    par_mV     counts    roll    pitch   yaw     "
    "wTemp   press     depth"
  ));
  Serial.println(F(
    "------------------------------------------------------------"
    "--------------------------------------------"
  ));

  // Set the sub-second reference just before the first sample.
  {
    DateTime t0 = rtc.now();
    rtcAnchorUnix   = t0.unixtime();
    rtcAnchorMillis = millis();
  }

  lastSample = millis();
}

// Integrates one IMU reading into the orientation estimate. Called from both
// setup() and loop() so the filter keeps running between logged samples.
void updateFilter() {
  uint32_t nowUs = micros();
  if (nowUs - lastFilterUpdate < FILTER_INTERVAL_US) return;
  lastFilterUpdate = nowUs;

  sensors_event_t accel, gyro, mag, temp;
  lsm6ds.getEvent(&accel, &gyro, &temp);
  lis3mdl.getEvent(&mag);

  float mx = (mag.magnetic.x - MAG_OFFSET_X) * MAG_SCALE_X;
  float my = (mag.magnetic.y - MAG_OFFSET_Y) * MAG_SCALE_Y;
  float mz = (mag.magnetic.z - MAG_OFFSET_Z) * MAG_SCALE_Z;

  // Madgwick expects gyro in deg/s and accel in g. The Adafruit library
  // returns rad/s and m/s2, so convert on the way in.
  filter.update(
    gyro.gyro.x * 57.29578,
    gyro.gyro.y * 57.29578,
    gyro.gyro.z * 57.29578,
    accel.acceleration.x / 9.80665,
    accel.acceleration.y / 9.80665,
    accel.acceleration.z / 9.80665,
    mx, my, mz
  );
}

void loop() {
  updateFilter();

  if (millis() - lastSample < SAMPLE_INTERVAL_MS) return;

  // Advance the schedule by one fixed interval to hold an even sample rate.
  // If a sample ran long, resync to the present instead, so the logger does
  // not fire a burst of back-to-back samples catching up.
  lastSample += SAMPLE_INTERVAL_MS;
  if (millis() - lastSample >= SAMPLE_INTERVAL_MS) {
    lastSample = millis();
    overrunCount++;
  }

  digitalWrite(LED_PIN, HIGH);

  // ---- PAR ----
  // Differential A0-A1 rejects noise common to both sensor leads.
  int32_t adsAccum = 0;
  for (uint8_t i = 0; i < PAR_OVERSAMPLE; i++) {
    adsAccum += ads.readADC_Differential_0_1();
  }
  float par_counts = (float)adsAccum / (float)PAR_OVERSAMPLE;
  float par_mV     = par_counts * ADS_LSB_MV;
  float par_uMol   = par_mV * SQ500_CAL_FACTOR;

  // ---- Depth, pressure, water temperature ----
  bar30.read();
  float depth_m       = bar30.depth();
  float pressure_mbar = bar30.pressure();
  float water_temp    = bar30.temperature();

  // ---- Raw IMU, logged alongside the fused angles so orientation can be
  //      recomputed later with different filter settings ----
  sensors_event_t accel, gyro, mag, imuTemp;
  lsm6ds.getEvent(&accel, &gyro, &imuTemp);
  lis3mdl.getEvent(&mag);

  float mx_cal = (mag.magnetic.x - MAG_OFFSET_X) * MAG_SCALE_X;
  float my_cal = (mag.magnetic.y - MAG_OFFSET_Y) * MAG_SCALE_Y;
  float mz_cal = (mag.magnetic.z - MAG_OFFSET_Z) * MAG_SCALE_Z;

  // ---- Fused orientation, used to screen samples by sensor tilt ----
  float roll  = filter.getRoll();
  float pitch = filter.getPitch();
  float yaw   = filter.getYaw();

  // ---- Timestamp ----
  DateTime now     = rtc.now();
  uint32_t ms      = millis();
  uint32_t nowUnix = now.unixtime();
  if (nowUnix != rtcAnchorUnix) {
    rtcAnchorUnix   = nowUnix;
    rtcAnchorMillis = ms;
  }
  uint32_t subSec = ms - rtcAnchorMillis;
  if (subSec > 999) subSec = 999;
  char isoTime[30];
  snprintf(isoTime, sizeof(isoTime),
           "%04d-%02d-%02dT%02d:%02d:%02d.%03lu",
           now.year(), now.month(), now.day(),
           now.hour(), now.minute(), now.second(),
           (unsigned long)subSec);

  // ---- CSV row ----
  // par_mV and par_uMol carry extra decimals because oversampling makes the
  // effective step ADS_LSB_MV / PAR_OVERSAMPLE.
  dataFile.print(isoTime);          dataFile.print(',');
  dataFile.print(now.unixtime());   dataFile.print(',');
  dataFile.print(par_mV, 6);        dataFile.print(',');
  dataFile.print(par_uMol, 4);      dataFile.print(',');
  dataFile.print(depth_m, 3);       dataFile.print(',');
  dataFile.print(pressure_mbar, 2); dataFile.print(',');
  dataFile.print(water_temp, 2);    dataFile.print(',');
  dataFile.print(accel.acceleration.x, 3); dataFile.print(',');
  dataFile.print(accel.acceleration.y, 3); dataFile.print(',');
  dataFile.print(accel.acceleration.z, 3); dataFile.print(',');
  dataFile.print(gyro.gyro.x, 4);   dataFile.print(',');
  dataFile.print(gyro.gyro.y, 4);   dataFile.print(',');
  dataFile.print(gyro.gyro.z, 4);   dataFile.print(',');
  dataFile.print(mx_cal, 2);        dataFile.print(',');
  dataFile.print(my_cal, 2);        dataFile.print(',');
  dataFile.print(mz_cal, 2);        dataFile.print(',');
  dataFile.print(roll, 2);          dataFile.print(',');
  dataFile.print(pitch, 2);         dataFile.print(',');
  dataFile.print(yaw, 2);           dataFile.print(',');
  dataFile.print(imuTemp.temperature, 2); dataFile.print(',');
  // millis_boot: free-running clock, gives exact spacing between samples.
  dataFile.print((unsigned long)ms);      dataFile.print(',');
  // par_adc_counts: mean raw ADC code, the unconverted digitizer output.
  dataFile.println(par_counts, 4);

  // Flush every 16 samples (~2 s) so a power loss costs at most 2 s of data,
  // without writing to the card on every row.
  sampleCount++;
  if (sampleCount % 16 == 0) {
    dataFile.flush();
  }

  // Serial preview at ~1 Hz, a readable subset for checking the instrument
  // before deployment.
  if (sampleCount % 8 == 0) {
    Serial.print(isoTime);            Serial.print(F("  "));
    printPadded(par_uMol,      2, 6);
    printPadded(par_mV,        4, 7);
    printPadded(par_counts,    2, 9);
    printPadded(roll,          2, 7);
    printPadded(pitch,         2, 7);
    printPadded(yaw,           2, 7);
    printPadded(water_temp,    2, 6);
    printPadded(pressure_mbar, 2, 8);
    Serial.println(depth_m, 3);
  }

  // Overrun count, reported once a minute: a rising number means the work
  // per sample no longer fits the sample interval.
  if (sampleCount % 480 == 0 && overrunCount > 0) {
    Serial.print(F("  [warning] timing overruns so far: "));
    Serial.println(overrunCount);
  }

  digitalWrite(LED_PIN, LOW);
}
