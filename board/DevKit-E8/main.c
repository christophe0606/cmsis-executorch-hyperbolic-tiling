/*---------------------------------------------------------------------------
 * Copyright (c) 2025-2026 Arm Limited (or its affiliates). All rights reserved.
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
 * DevKit-E8 (M55_HP) board bring-up for a headless Ethos-U85 runner. Derived
 * from the pack's Boards/DevKit-e8/Layers/M55_HP/main.c without the MIPI,
 * USB, Ethernet and VIO initialisation.
 *---------------------------------------------------------------------------*/

#include "RTE_Components.h"
#include CMSIS_device_header

#include "board_config.h"
#include "main.h"

#include "se_services_port.h"
#include "board_display.h"

/* VBAT power control bits for the MIPI TX DPHY and its PLL (pack layer main.c) */
#define VBAT_PWR_CTRL_TX_DPHY_PWR_MASK        (1U <<  0) /* Mask off the power supply for MIPI TX DPHY */
#define VBAT_PWR_CTRL_TX_DPHY_ISO             (1U <<  1) /* Enable isolation for MIPI TX DPHY */
#define VBAT_PWR_CTRL_RX_DPHY_PWR_MASK        (1U <<  4) /* Mask off the power supply for MIPI RX DPHY */
#define VBAT_PWR_CTRL_RX_DPHY_ISO             (1U <<  5) /* Enable isolation for MIPI RX DPHY */
#define VBAT_PWR_CTRL_DPHY_PLL_PWR_MASK       (1U <<  8) /* Mask off the power supply for MIPI PLL */
#define VBAT_PWR_CTRL_DPHY_PLL_ISO            (1U <<  9) /* Enable isolation for MIPI PLL */
#define VBAT_PWR_CTRL_DPHY_VPH_1P8_PWR_BYP_EN (1U << 12) /* dphy vph 1p8 power bypass enable */

/*
  Power up the MIPI DPHY (the display's physical layer), as the pack's
  DevKit-e8 layer does in its vbat_init().
*/
static void dphy_power_init(void)
{
    VBAT->PWR_CTRL &= ~(VBAT_PWR_CTRL_TX_DPHY_PWR_MASK | VBAT_PWR_CTRL_RX_DPHY_PWR_MASK |
                        VBAT_PWR_CTRL_DPHY_PLL_PWR_MASK | VBAT_PWR_CTRL_DPHY_VPH_1P8_PWR_BYP_EN);
    VBAT->PWR_CTRL &= ~(VBAT_PWR_CTRL_TX_DPHY_ISO | VBAT_PWR_CTRL_RX_DPHY_ISO | VBAT_PWR_CTRL_DPHY_PLL_ISO);
}

int main(void)
{
    /* Apply the Conductor pin configuration (includes the UART4 console pins) */
    board_pins_config();

    /* Apply the Conductor GPIO configuration */
    board_gpios_config();

    /* Bring up the Secure Enclave services (MHU link to the SE) */
    se_services_port_init();

    /* Request the clocks the SE has to enable for this core */
    board_clocks_config(CLKEN_HFOSC_MASK | CLKEN_CLK_100M_MASK);

    /* Power up the MIPI DPHY before the display driver touches it */
    dphy_power_init();

    /* Initialize STDIO (UART4 on the PRG USB connector) */
    if (stdio_init() != 0) return 1;

    #if defined(ETHOSU_ARCH)
    /* Initialize Ethos NPU */
    ethos_setup();
    #endif

    return app_main();
}
