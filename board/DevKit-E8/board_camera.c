// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "RTE_Components.h"
#include CMSIS_device_header
#include "RTE_Device.h"
#include "Driver_CPI.h"
#include "board_camera.h"
#include "../../src/camera_texture.h"

#if RTE_MT9M114_CAMERA_SENSOR_MIPI_IMAGE_CONFIG != 5
#error "Video texture requires the MT9M114 320x320 RGB565 configuration"
#endif
#if RTE_CPI_STREAMING_ENABLE || RTE_CPI_NUM_ACTIVE_FRAMEBUFFERS != 0
#error "Video texture uses one CPI snapshot buffer, not the streaming pointer-array API"
#endif

extern ARM_DRIVER_CPI Driver_CPI;
enum { CAMERA_WIDTH = 320 };
// SRAM0 has room beside the NPU scratch pool; SRAM1 holds the LCD and HE mailbox.
static uint16_t camera_frame[CAMERA_WIDTH * CAMERA_WIDTH]
    __attribute__((section(".bss.camera_frame_buf"), aligned(32)));
static volatile uint32_t camera_events;
static uint32_t capture_started;
static int initialized, configured, active;
static int32_t stop_error;
static const uint32_t error_events = ARM_CPI_EVENT_MIPI_CSI2_ERROR |
    ARM_CPI_EVENT_ERR_CAMERA_INPUT_FIFO_OVERRUN |
    ARM_CPI_EVENT_ERR_CAMERA_OUTPUT_FIFO_OVERRUN | ARM_CPI_EVENT_ERR_HARDWARE;

static void camera_callback(uint32_t events) { camera_events |= events; }

void camera_stop(void) {
  if (!active) return;
  stop_error = Driver_CPI.Stop();
  // Never reuse a buffer after a failed stop: a reset is then required.
  active = 0;
  if (stop_error == ARM_DRIVER_OK) {
    camera_events = 0;
  }
}

static int32_t camera_start_frame(void) {
  camera_events = 0;
  // No CPU writes are allowed while CPI owns this aligned DMA buffer.
  SCB_CleanInvalidateDCache_by_Addr(camera_frame, sizeof(camera_frame));
  __DSB();
  capture_started = DWT->CYCCNT;
  active = 1; // Also stop on a partially failed CaptureFrame.
  return Driver_CPI.CaptureFrame(camera_frame);
}

int32_t camera_update_texture(int8_t *texture, unsigned size) {
  int32_t status;
  if (!texture || !size || size > CAMERA_WIDTH) return ARM_DRIVER_ERROR_PARAMETER;
  if (stop_error != ARM_DRIVER_OK) return stop_error;
  if (!initialized) {
    status = Driver_CPI.Initialize(camera_callback);
    if (status != ARM_DRIVER_OK) return status;
    initialized = 1;
  }
  if (!configured) {
    // Same bring-up order as ModelNova's vStream VideoIn, using the pack drivers.
    status = Driver_CPI.PowerControl(ARM_POWER_FULL);
    if (status != ARM_DRIVER_OK) return status;
    active = 1;
    status = Driver_CPI.Stop();
    if (status != ARM_DRIVER_OK) return status;
    active = 0;
    status = Driver_CPI.Control(CPI_CAMERA_SENSOR_CONFIGURE, 0);
    if (status != ARM_DRIVER_OK) return status;
    status = Driver_CPI.Control(CPI_CONFIGURE, 0);
    if (status != ARM_DRIVER_OK) return status;
    configured = 1;
  }
  if (!active) {
    status = Driver_CPI.Control(CPI_CONFIGURE, 0);
    if (status != ARM_DRIVER_OK) return status;
    // Stop disables the events. Re-enable them each time video mode is entered.
    status = Driver_CPI.Control(CPI_EVENTS_CONFIGURE,
        ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED | error_events);
    if (status != ARM_DRIVER_OK) return status;
    status = camera_start_frame();
    return status == ARM_DRIVER_OK ? 0 : status;
  }
  const uint32_t events = camera_events;
  if (events & error_events) return ARM_DRIVER_ERROR;
  if (!(events & ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED)) {
    // At 800 MHz the unsigned subtraction remains valid across a counter wrap.
    if ((uint32_t)(DWT->CYCCNT - capture_started) > SystemCoreClock * 2U)
      return ARM_DRIVER_ERROR_TIMEOUT;
    return 0;
  }
  __DSB();
  SCB_InvalidateDCache_by_Addr(camera_frame, sizeof(camera_frame));
  __DSB();
  camera_rgb565_to_texture(camera_frame, CAMERA_WIDTH, texture, size);
  // Snapshot DMA is stopped now. Start the next capture only after copying;
  // rendering uses the separate texture snapshot while CPI fills this buffer.
  status = camera_start_frame();
  return status == ARM_DRIVER_OK ? 1 : status;
}
