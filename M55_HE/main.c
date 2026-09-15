/* Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */

extern void tiling_he_worker(void);

int main(void)
{
    /* Only the Helium worker runs here. HP owns Ethos, UART and display. */
    tiling_he_worker();
}
