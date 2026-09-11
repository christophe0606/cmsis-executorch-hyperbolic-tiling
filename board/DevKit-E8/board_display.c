/*---------------------------------------------------------------------------
 * Copyright (c) 2026 Arm Limited (or its affiliates). All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * DevKit-E8 display bring-up through the pack's CDC200 driver, the sequence
 * the pack's vStream VideoOut driver uses: Initialize, PowerControl, enable
 * the start-of-frame event, configure the display (which also initialises
 * the MIPI DSI link and the ILI9806E panel), set a frame buffer, Start.
 *---------------------------------------------------------------------------*/

#include "RTE_Components.h"
#include CMSIS_device_header

#include "Driver_CDC200.h"
#include "board_display.h"

extern ARM_DRIVER_CDC200 Driver_CDC200;
static ARM_DRIVER_CDC200 *cdc = &Driver_CDC200;

static volatile uint32_t frames_started;

static void cdc_callback(uint32_t event)
{
    if (event & ARM_CDC_SCANLINE0_EVENT) {
        frames_started++;
    }
}

int32_t display_init(void)
{
    int32_t status;

    status = cdc->Initialize(cdc_callback);
    if (status != ARM_DRIVER_OK) {
        return status;
    }
    status = cdc->PowerControl(ARM_POWER_FULL);
    if (status != ARM_DRIVER_OK) {
        return status;
    }
    status = cdc->Control(CDC200_SCANLINE0_EVENT, 1U);
    if (status != ARM_DRIVER_OK) {
        return status;
    }
    return cdc->Control(CDC200_CONFIGURE_DISPLAY, 0U);
}

int32_t display_start(const void *fb)
{
    int32_t status = cdc->Control(CDC200_FRAMEBUF_UPDATE, (uint32_t)fb);
    if (status != ARM_DRIVER_OK) {
        return status;
    }
    return cdc->Start();
}

int32_t display_present(const void *fb)
{
    return cdc->Control(CDC200_FRAMEBUF_UPDATE_VSYNC, (uint32_t)fb);
}

uint32_t display_frame_count(void)
{
    return frames_started;
}

int32_t display_wait_frame(uint32_t count)
{
    /* A 60 Hz frame is 16.7 ms; give up after roughly four of them. */
    for (uint32_t spins = 0; spins < 4000000U; ++spins) {
        if ((int32_t)(frames_started - count) > 0) {
            return 0;
        }
    }
    return -1;
}
