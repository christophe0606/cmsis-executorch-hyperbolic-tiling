// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tiling_settings.hpp"

void mcp_tools_init(Settings& settings, const volatile uint32_t* frames = nullptr);
// Renderer invalidations: 1 = symmetry, 2 = maps, 4 = colours.
int mcp_take_changes();
