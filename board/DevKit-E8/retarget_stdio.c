/*---------------------------------------------------------------------------
 * Copyright (c) 2025 Arm Limited (or its affiliates).
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *      Name:    retarget_stdio.c
 *      Purpose: Retarget stdio to CMSIS UART
 *
 *---------------------------------------------------------------------------*/

#include "RTE_Components.h"
#include CMSIS_target_header
#include CMSIS_device_header
#include "board_console.h"

/* Compile-time configuration */
#ifndef UART_BAUDRATE
#define UART_BAUDRATE 115200
#endif

/* Exported function */
extern int stdio_init(void);

/* Reference to the underlying USART driver */
#define ptrUSART (&ARM_Driver_USART_(RETARGET_STDIO_UART))

/* One ISR producer, one foreground consumer. No parsing or printing in IRQ. */
#define RX_SIZE 8192U
static uint8_t rx_ring[RX_SIZE];
static uint8_t rx_byte;
static volatile uint32_t rx_write, rx_read, rx_lost;
static int initialized;

static void uart_event(uint32_t event)
{
    const uint32_t errors = ARM_USART_EVENT_RX_OVERFLOW | ARM_USART_EVENT_RX_BREAK |
                            ARM_USART_EVENT_RX_FRAMING_ERROR | ARM_USART_EVENT_RX_PARITY_ERROR;
    if (event & errors) {
        rx_lost = 1U;
        ptrUSART->Control(ARM_USART_ABORT_RECEIVE, 0U);
    } else if (event & ARM_USART_EVENT_RECEIVE_COMPLETE) {
        uint32_t next = (rx_write + 1U) % RX_SIZE;
        if (next == rx_read) {
            rx_lost = 1U;
        } else if (!rx_lost) {
            rx_ring[rx_write] = rx_byte;
            __DMB();
            rx_write = next;
        }
    } else {
        return; /* TX completion or RX timeout: keep the outstanding receive. */
    }
    if (ptrUSART->Receive(&rx_byte, 1U) != ARM_DRIVER_OK) rx_lost = 1U;
}

int board_console_getchar(void)
{
    uint32_t state = __get_PRIMASK();
    __disable_irq();
    int ch = -1;
    if (rx_lost) {
        rx_read = rx_write;
        rx_lost = 0U;
        ch = -2;
    } else if (rx_read != rx_write) {
        ch = rx_ring[rx_read];
        rx_read = (rx_read + 1U) % RX_SIZE;
    }
    __set_PRIMASK(state);
    return ch;
}

/* Foreground-only synchronous writes over the interrupt-driven CMSIS driver.
 * Waiting for tx_busy to clear also keeps the buffer alive until completion.
 */
int stdout_putchar(int ch)
{
    uint8_t byte = (uint8_t)ch;
    if (!initialized || ptrUSART->Send(&byte, 1U) != ARM_DRIVER_OK) return -1;
    while (ptrUSART->GetStatus().tx_busy) {}
    return ch;
}

int stderr_putchar(int ch) { return stdout_putchar(ch); }
int stdin_getchar(void)
{
    int ch;
    do { ch = board_console_getchar(); } while (ch == -1);
    return ch < 0 ? -1 : ch;
}

/**
  Initialize stdio

  \return          0 on success, or -1 on error.
*/
int stdio_init(void)
{

    if (initialized) return 0;
    if (ptrUSART->Initialize(uart_event) != ARM_DRIVER_OK) {
        return -1;
    }

    if (ptrUSART->PowerControl(ARM_POWER_FULL) != ARM_DRIVER_OK) {
        return -1;
    }

    if (ptrUSART->Control(ARM_USART_MODE_ASYNCHRONOUS | ARM_USART_DATA_BITS_8 |
                              ARM_USART_PARITY_NONE | ARM_USART_STOP_BITS_1 |
                              ARM_USART_FLOW_CONTROL_NONE,
                          UART_BAUDRATE) != ARM_DRIVER_OK) {
        return -1;
    }

#if defined(RTE_CMSIS_Compiler_STDIN_Custom)
    if (ptrUSART->Control(ARM_USART_CONTROL_RX, 1U) != ARM_DRIVER_OK) {
        return -1;
    }
#endif

#if defined(RTE_CMSIS_Compiler_STDERR_Custom) || defined(RTE_CMSIS_Compiler_STDOUT_Custom)
    if (ptrUSART->Control(ARM_USART_CONTROL_TX, 1U) != ARM_DRIVER_OK) {
        return -1;
    }
#endif

    /* The first receive is armed before returning; later ones rearm in IRQ. */
    if (ptrUSART->Receive(&rx_byte, 1U) != ARM_DRIVER_OK) return -1;
    initialized = 1;
    return 0;
}
