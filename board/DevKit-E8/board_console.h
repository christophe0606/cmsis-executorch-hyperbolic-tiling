// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#ifndef BOARD_CONSOLE_H
#define BOARD_CONSOLE_H
#ifdef __cplusplus
extern "C" {
#endif
// Nonblocking: byte, -1 if empty, -2 if input was lost (resync at newline).
int board_console_getchar(void);
#ifdef __cplusplus
}
#endif
#endif
