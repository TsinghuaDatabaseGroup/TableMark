import numpy as np
import os
import random
import time
import heapq
import json
from numba import njit
from pyroaring import BitMap64

from globals import *


def hamming_distance(a: int, b: int) -> int:
  return (a ^ b).bit_count()


bit_masks = np.asarray([(1 << i) for i in range(32)], dtype=int)
@njit
def get_neighbors(watermark_int: int) -> np.ndarray:
  return watermark_int ^ bit_masks


lookup_16bit = np.asarray([i.bit_count() for i in range(1 << 16)], dtype=np.uint8)
@njit(boundscheck=False)
def all_dist32(code: int, codes1: np.ndarray, codes2: np.ndarray, d: int):
  return (lookup_16bit[(code >> 16) ^ codes1] + lookup_16bit[(code & 0xFFFF) ^ codes2] >= d).all()


def gen_general_bfs(n: int, m: int, d: int) -> tuple[list, list]:
  codes = []
  losses = []
  visited = set()
  to_visit = [(0, (np.uint32 if m <= 32 else np.uint64 if m <= 64 else '?')(0))]
  last_time = time.time_ns() / 1e9
  print('Begin General BFS', flush=True)
  while to_visit:
    loss, code = heapq.heappop(to_visit)
    if all(hamming_distance(code, c) >= d for c in codes):
      codes.append(code)
      losses.append(loss)
      if len(codes) % 10 == 0:
        now = time.time_ns() / 1e9
        print(f'Generated: {len(codes):5d} / {n}, Loss: {loss:9.2f}, Explored: {len(visited) - len(to_visit):9d}, To Explore: {len(to_visit):9d}, Time: {now - last_time:9.3f} s', flush=True)
        last_time = now
        if len(codes) >= n:
          break
    for nb in get_neighbors(code):
      if nb not in visited:
        visited.add(nb)
        heapq.heappush(to_visit, (int(nb).bit_count(), nb))
  return codes, losses


def gen_general_bfs32(n: int, d: int) -> tuple[list, list]:
  codes = [0]
  losses = []
  codes_numpy = np.zeros(shape=(2, n), dtype=np.uint16, order='C')
  codes_numpy1 = codes_numpy[0]
  codes_numpy2 = codes_numpy[1]
  visited = BitMap64()
  visited.add(0)
  to_visit = [0]
  last_time = time.time_ns() / 1e9
  print('Begin General BFS', flush=True)
  while to_visit:
    tp = heapq.heappop(to_visit)
    code = tp & 0xFFFFFFFF
    if all_dist32(code, codes_numpy1, codes_numpy2, d):
      codes_numpy1[len(codes)] = code >> 16
      codes_numpy2[len(codes)] = code & 0xFFFF
      codes.append(code)
      loss = tp >> 32
      losses.append(loss)
      if len(codes) % 10 == 0:
        now = time.time_ns() / 1e9
        print(f'Generated: {len(codes):5d} / {n}, Loss: {loss:9.2f}, Explored: {len(visited) - len(to_visit):9d}, To Explore: {len(to_visit):9d}, Time: {now - last_time:9.3f} s', flush=True)
        last_time = now
        if len(codes) >= n:
          break
    for nb in get_neighbors(code):
      if nb not in visited:
        visited.add(nb)
        heapq.heappush(to_visit, (nb.bit_count() << 32) + nb)
  return codes, losses


def gen_general(code_path: str, loss_path: str, loss: str, n: int, m: int, d: int):
  start = time.time_ns()
  if loss.__contains__('bfs'):
    if m == 32:
      codes, losses = gen_general_bfs32(n, d)
    else:
      codes, losses = gen_general_bfs(n, m, d)
  else:
    assert False, loss
  end = time.time_ns()
  dur = (end - start) / 1e9
  assert len(codes) == CFG_WATERMARK.NUM_USERS, f'Too Large Min Hamming Distance {CFG_WATERMARK.MIN_HAMMING_DIST} !'
  min_hamming_dist = min(hamming_distance(codes[i], codes[j]) for i in range(len(codes)) for j in range(i + 1, len(codes)))
  assert min_hamming_dist == CFG_WATERMARK.MIN_HAMMING_DIST, f'{min_hamming_dist} != {CFG_WATERMARK.MIN_HAMMING_DIST}'
  print(f'Verified: min hanmming distance = {min_hamming_dist}')
  with open(code_path, 'w') as f:
    for code in codes:
      f.write(str(code) + '\n')
  with open(loss_path, 'w') as f:
    for loss in losses:
      f.write(str(loss) + '\n')
  with open(f'{CFG_BASIC.ROOT_DIR}/watermark/results_final/gen_code.json', 'a') as f:
    f.write(json.dumps({**get_res_header(CFG_BASIC.SEED), 'duration': dur}) + '\n')


def main():
  assert isinstance(CFG_WATERMARK.MIN_HAMMING_DIST, int) and CFG_WATERMARK.MIN_HAMMING_DIST > 0
  assert not os.path.exists(CFG_WATERMARK.CODE_PATH), 'Already Exists !'
  os.makedirs(os.path.dirname(CFG_WATERMARK.CODE_PATH), exist_ok=True)

  if CFG_WATERMARK.GEN_CODE_LOSS.__contains__('general'):
    return gen_general(CFG_WATERMARK.CODE_PATH, CFG_WATERMARK.CODE_LOSS_PATH, CFG_WATERMARK.GEN_CODE_LOSS, CFG_WATERMARK.NUM_USERS, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MIN_HAMMING_DIST)

  assert CFG_WATERMARK.GEN_CODE_LOSS == 'none'
  assert isinstance(CFG_WATERMARK.MIN_HAMMING_DIST, int)
  watermark_ints = []

  while len(watermark_ints) < CFG_WATERMARK.NUM_USERS:
    watermark_int = random.randint(0, 2 ** CFG_WATERMARK.NUM_WATERMARK_BITS - 1)
    if all(hamming_distance(watermark_int, watermark_ints[i]) >= CFG_WATERMARK.MIN_HAMMING_DIST for i in range(len(watermark_ints))):
      watermark_ints.append(watermark_int)
      if len(watermark_ints) % 100 == 0:
        print(f'Generated: {len(watermark_ints)} / {CFG_WATERMARK.NUM_USERS}', flush=True)
    
  assert len(watermark_ints) == CFG_WATERMARK.NUM_USERS, f'Too Large Min Hamming Distance {CFG_WATERMARK.MIN_HAMMING_DIST} !'
  min_hamming_dist = min(hamming_distance(watermark_ints[i], watermark_ints[j]) for i in range(len(watermark_ints)) for j in range(i + 1, len(watermark_ints)))
  assert min_hamming_dist >= CFG_WATERMARK.MIN_HAMMING_DIST, f'{min_hamming_dist} != {CFG_WATERMARK.MIN_HAMMING_DIST}'
  print(f'Verified: min hanmming distance = {min_hamming_dist}')
  np.savetxt(CFG_WATERMARK.CODE_PATH, watermark_ints, '%d')
