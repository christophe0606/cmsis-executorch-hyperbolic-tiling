// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tiling_kernel.hpp"

namespace tiling {
// Single producer/single consumer: HP owns all scheduling and output routing.
// Only one strip is in flight on HE. Inputs remain immutable until completion.
class WorkerClient {
 public:
  bool init();
  void begin_frame(const RenderState& state, float time);
  void submit(int bx, int by);
  bool ready();
  bool receive(uint16_t* accum, GeometryStats& stats);
  bool wait(uint16_t* accum, GeometryStats& stats);
  bool available() const { return available_; }
  uint32_t worker_clock() const;
 private:
  uint32_t sequence_ = 0, frame_ = 0, started_ = 0;
  float time_ = 0;
  bool available_ = false;
  void request(uint32_t opcode, int bx, int by);
};
[[noreturn]] void worker_main();
} // namespace tiling
