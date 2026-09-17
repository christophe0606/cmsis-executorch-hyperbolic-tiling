// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "camera_texture.h"
#include <array>
#include <cstdio>

int main() {
  // Only the sample centers contain colour. The other pixels must be skipped.
  const uint16_t frame[16] = {0, 0, 0, 0, 0, 0xf800, 0, 0x07e0,
                             0, 0, 0, 0, 0, 0x001f, 0, 0xffff};
  std::array<int8_t, 14> output{};
  output.front() = 42;
  output.back() = 43;
  camera_rgb565_to_texture(frame, 4, output.data() + 1, 2);
#if defined(APP_DISPLAY_BGR) && APP_DISPLAY_BGR
  // LCD byte order: a red camera pixel must populate the third plane.
  const std::array<int8_t, 14> expected = {42, -128, -128, 127, 127,
                                         -128, 127, -128, 127,
                                         127, -128, -128, 127, 43};
#else
  const std::array<int8_t, 14> expected = {42, 127, -128, -128, 127,
                                         -128, 127, -128, 127,
                                         -128, -128, 127, 127, 43};
#endif
  if (output != expected) {
    std::fprintf(stderr, "RGB565 channel order, quantization, sampling or bounds failed\n");
    return 1;
  }
  const uint16_t middle = 0x8410; // R=16/31, G=32/63, B=16/31.
  int8_t rgb[3];
  camera_rgb565_to_texture(&middle, 1, rgb, 1);
  if (rgb[0] != 4 || rgb[1] != 2 || rgb[2] != 4) return 2;
  return 0;
}
