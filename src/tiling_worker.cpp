// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "tiling_worker.hpp"
#include <cstring>
#include <type_traits>
#include "RTE_Components.h"
#include CMSIS_device_header
#include "app_mem_regions.h"

namespace tiling {
namespace {
// Version 2 carries the AA enum and center rectangle in RenderState.
constexpr uint32_t kMagic = 0x54494c45, kVersion = 2;
constexpr uint32_t kPing = 1, kRender = 2, kOk = 0, kInvalid = 1;
struct alignas(32) Request {
  uint32_t magic, version, sequence, opcode, frame;
  int32_t bx, by;
  float time;
};
struct alignas(32) Response {
  uint32_t sequence, version, status, clock;
  GeometryStats stats;
  uint32_t cycles;
};
struct alignas(32) Shared {
  Request request;
  Response response;
  RenderState state;
  alignas(32) uint16_t accum[3 * kPixels];
};
static_assert(sizeof(Request) == 32 && sizeof(Response) == 32);
static_assert(sizeof(Shared) <= APP_TILING_SHARED_SIZE);
static_assert(std::is_trivially_copyable<RenderState>::value);
static_assert(sizeof(RenderState) % 32 == 0);
Shared& shared = *reinterpret_cast<Shared*>(APP_TILING_SHARED_BASE);

// This SRAM is cacheable on both cores. Each cache line has exactly one writer.
// Publish payload first, then sequence; invalidate before consuming remote data.
void clean(void* address, size_t bytes) {
  SCB_CleanDCache_by_Addr(address, static_cast<int32_t>(bytes));
  __DSB();
}
void invalidate(void* address, size_t bytes) {
  SCB_InvalidateDCache_by_Addr(address, static_cast<int32_t>(bytes));
  __DSB();
}
uint32_t response_sequence() {
  invalidate(&shared.response, sizeof(Response));
  const uint32_t seq = reinterpret_cast<volatile Response&>(shared.response).sequence;
  __DMB();
  return seq;
}
uint32_t next(uint32_t sequence) { return sequence + 1 ? sequence + 1 : 1; }
} // namespace

void WorkerClient::request(uint32_t opcode, int bx, int by) {
  sequence_ = next(sequence_);
  volatile Request& req = shared.request;
  req.magic = kMagic;
  req.version = kVersion;
  req.opcode = opcode;
  req.frame = frame_;
  req.bx = bx;
  req.by = by;
  req.time = time_;
  __DMB();
  req.sequence = sequence_;
  clean(&shared.request, sizeof(Request));
  started_ = DWT->CYCCNT;
}

bool WorkerClient::init() {
  // Ping first: wait for any request left by an earlier HP session to finish
  // before overwriting shared frame inputs. The old idle HE image times out.
  invalidate(&shared.request, sizeof(Request));
  sequence_ = shared.request.sequence;
  request(kPing, 0, 0);
  while (!ready()) {
    if (static_cast<uint32_t>(DWT->CYCCNT - started_) > SystemCoreClock / 2) return false;
  }
  available_ = shared.response.version == kVersion && shared.response.status == kOk;
  return available_;
}

void WorkerClient::begin_frame(const RenderState& state, float time) {
  if (!available_) return;
  std::memcpy(&shared.state, &state, sizeof(state));
  clean(&shared.state, sizeof(state));
  frame_ = next(sequence_);
  time_ = time;
}

void WorkerClient::submit(int bx, int by) { request(kRender, bx, by); }
bool WorkerClient::ready() { return response_sequence() == sequence_; }
uint32_t WorkerClient::worker_clock() const { return shared.response.clock; }

bool WorkerClient::receive(uint16_t* accum, GeometryStats& stats) {
  if (!ready()) return false;
  if (shared.response.version != kVersion || shared.response.status != kOk) {
    available_ = false;
    return false;
  }
  invalidate(shared.accum, sizeof(shared.accum));
  std::memcpy(accum, shared.accum, sizeof(shared.accum));
  stats = shared.response.stats;
  return true;
}

bool WorkerClient::wait(uint16_t* accum, GeometryStats& stats) {
  while (!ready()) {
    if (static_cast<uint32_t>(DWT->CYCCNT - started_) > SystemCoreClock * 2U) {
      // No further requests or input writes after timeout. A late HE result
      // stays isolated in the mailbox; HP can safely recompute the strip.
      available_ = false;
      return false;
    }
  }
  return receive(accum, stats);
}

[[noreturn]] void worker_main() {
  static RenderState local_state;
  alignas(32) static uint16_t local_accum[3 * kPixels];
  uint32_t last_sequence = 0, last_frame = 0;
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  for (;;) {
    invalidate(&shared.request, sizeof(Request));
    const auto& source = reinterpret_cast<volatile Request&>(shared.request);
    const uint32_t seq = source.sequence;
    if (!seq || seq == last_sequence || source.magic != kMagic || source.version != kVersion) {
      for (int i = 0; i < 256; ++i) __NOP();
      continue;
    }
    __DMB();
    const uint32_t opcode = source.opcode, frame = source.frame;
    const int bx = source.bx, by = source.by;
    const float time = source.time;
    GeometryStats stats{};
    uint32_t status = kOk;
    const uint32_t start = DWT->CYCCNT;
    if (opcode == kPing) {
      last_frame = 0;
    } else if (opcode == kRender) {
      if (frame != last_frame) {
        invalidate(&shared.state, sizeof(RenderState));
        std::memcpy(&local_state, &shared.state, sizeof(RenderState));
        last_frame = frame;
      }
      const int width = local_state.settings.half ? kFrameWidth / 2 : kFrameWidth;
      const int height = local_state.settings.half ? kFrameHeight / 2 : kFrameHeight;
      if (bx < 0 || bx + kWidth > width || by < -1 || by >= height ||
          local_state.settings.iterations < 1 || local_state.settings.iterations > 40) {
        status = kInvalid;
      } else {
        stats = render_strip(local_state, local_accum, time, bx, by);
        std::memcpy(shared.accum, local_accum, sizeof(local_accum));
        clean(shared.accum, sizeof(shared.accum));
      }
    } else {
      status = kInvalid;
    }
    shared.response.version = kVersion;
    shared.response.status = status;
    shared.response.clock = SystemCoreClock;
    shared.response.stats = stats;
    shared.response.cycles = DWT->CYCCNT - start;
    __DMB();
    reinterpret_cast<volatile Response&>(shared.response).sequence = seq;
    clean(&shared.response, sizeof(Response));
    last_sequence = seq;
  }
}
} // namespace tiling

extern "C" void tiling_he_worker(void) { tiling::worker_main(); }
