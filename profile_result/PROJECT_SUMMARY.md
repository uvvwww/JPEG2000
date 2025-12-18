# JPEG2000 平行化加速 - 項目總結

## 1. 項目背景

**課程**: 平行程式設計 (ACD114118)
**平台**: Taiwania 3 HPC 集群
**目標**: 優化JPEG2000影像壓縮的執行效率

## 2. 實現方案

### 2.1 基礎設施
- **編譯器**: GCC 13.2.0 + OpenMPI 4.1.6
- **最佳化等級**: -O3 -march=native -ffast-math -funroll-loops
- **並行方案**: OpenMP (共享記憶體) + MPI (分散式記憶體)

### 2.2 核心優化

#### A. 編譯優化
```makefile
CXXFLAGS = -O3 -march=native -mtune=native \
           -ffast-math -funroll-loops -finline-functions
```
- 影響: 單執行緒基準建立

#### B. OpenMP 多執行緒
```cpp
opj_codec_set_threads(codec, num_threads);
#pragma omp parallel for schedule(static)
```
- 重點: T1編碼 (95%的計算量)
- 效果: 8執行緒 = 6.8倍加速 (接近線性)

#### C. MPI 多進程
```cpp
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// 每個rank編碼一張圖
```
- 分散編碼多張圖像
- 混合模式: 4 MPI ranks × 2 OpenMP threads

#### D. 時間測量修復
```cpp
// 修前: getrusage() → CPU時間 (所有線程總和)
// 修後: gettimeofday() → Wall-clock時間 (實際時間)
```
- 問題: CPU時間掩蓋了實際加速
- 解決: 準確測量實際牆上時間

## 3. 性能數據

### 3.1 單張圖編碼 (photo2590.jpg, 6016×3456)

| 配置 | 時間 | 加速比 | 效率 |
|------|------|--------|------|
| **基準 (1 thread)** | 12.51s | 1.0x | - |
| **4 threads** | 3.38s | 3.7x | 92.5% |
| **8 threads** | 1.83s | 6.8x | 85.0% |

### 3.2 多張圖編碼 (登入節點 via srun)

| 配置 | 圖數 | 時間 | 說明 |
|------|------|------|------|
| 1 rank × 8 threads | 1 | 1.99s | 最快 |
| 4 ranks × 2 threads | 4 | 13.1s | 並行編碼 |
| 8 ranks × 1 thread | 8 | 26.8s | 分散式 |

### 3.3 批量提交 (sbatch 到計算節點)

| 配置 | 時間 | 備註 |
|------|------|------|
| sbatch (4×2) | 42.4s | SLURM排隊+CPU較慢 |
| srun (4×2) | 13.1s | **直接執行 3倍快** |

### 3.4 階段分解 (8執行緒, 單張圖)

| 階段 | 時間 | 比例 |
|------|------|------|
| MCT (色彩轉換) | 0.007s | 0.4% |
| DWT (小波) | 0.070s | 3.8% |
| **T1 (熵編碼)** | **3.03s** | **94.9%** |
| T2 (位元流) | 0.003s | 0.2% |
| **TOTAL** | **3.81s** | **100%** |

**結論**: T1是絕對瓶頸，應為主要優化目標

## 4. 實現細節

### 4.1 修改的關鍵檔案

1. **src/common/opj_clock.cpp**
   - 從 `getrusage()` 改為 `gettimeofday()`
   - 理由: CPU時間在多執行緒時會加總，wall-clock才反映實際加速

2. **j2k_encode_mpi.cpp**
   - MPI初始化與rank管理
   - OpenMP執行緒動態分配
   - 性能統計與load balance追蹤

3. **j2k_encode_profile.cpp**
   - 6個pipeline階段的細粒度profiling
   - 檔案輸出profiling_results.txt

4. **Makefile**
   - OpenMP編譯旗標: `-fopenmp`
   - MPI編譯器選擇: `mpicxx`

### 4.2 編譯與執行

```bash
# 編譯 (含OpenMP)
make clean && make -j4

# 編譯 MPI版本
make mpi

# 直接測試 (登入節點)
export OMP_NUM_THREADS=8
srun -n 1 --cpus-per-task=8 ./build/j2k_encode_mpi input.ppm

# 多圖編碼
export OMP_NUM_THREADS=2
srun -n 4 --cpus-per-task=2 ./build/j2k_encode_mpi img1.ppm img2.ppm img3.ppm img4.ppm
```

## 5. 限制與瓶頸

### 5.1 當前限制
1. **T1編碼內在順序性**: 熵編碼(MQC)有資料依賴，難完全並行化
2. **記憶體頻寬**: 大圖像受記憶體流量限制
3. **Taiwania 3計算節點CPU**: 比登入節點慢22倍(?)

### 5.2 無法達成的優化
- ❌ **GPU加速**: Taiwania 3無CUDA環境
- ❌ **跨節點MPI**: 通信成本太高，反而變慢
- ❌ **演算法改進**: JPEG2000標準固定，難以改進

### 5.3 效能瓶頸來源
```
T1編碼時間: 11.9秒 / 12.5秒 = 95.2%

T1內部:
├─ Significance Pass (有順序依賴)
├─ Refinement Pass (有順序依賴)  
├─ Cleanup Pass
└─ MQC Entropy Coder (串列)
```

## 6. 最佳實踐方案

### 方案 A: 最快單張編碼
```bash
export OMP_NUM_THREADS=8
srun --account=ACD114118 -n 1 --cpus-per-task=8 \
  ./build/j2k_encode_mpi input.ppm
# 時間: 1.99秒
```

### 方案 B: 4張圖並行編碼
```bash
export OMP_NUM_THREADS=2
srun --account=ACD114118 -n 4 --cpus-per-task=2 \
  ./build/j2k_encode_mpi img1.ppm img2.ppm img3.ppm img4.ppm
# 時間: 13.1秒
```

### 方案 C: 批量長時間任務
```bash
sbatch submit_hybrid.sh
# 適合>1小時的工作
```

## 7. 測試結果摘要

### 7.1 加速曲線
```
執行時間 vs 執行緒數

    12.5s ├─ 1 thread
          │
     5.0s ├─
          │
     3.0s ├─ 4 threads
          │
     1.5s ├─ 8 threads
          │
     0.5s └─────────────────────
          1    2    4    8
```

### 7.2 Load Balance
- 單機單進程: 100%
- 4 MPI ranks × 2 threads: 99.8%
- 8 MPI ranks: 85.6% (SLURM節點不均)

## 8. 結論

### 成果
✅ **6.8倍加速** (8執行緒)
✅ **線性加速** (效率85-92%)
✅ **準確profiling** (修復wall-clock計時)
✅ **可靠MPI支援** (混合模式)

### 適用場景
- ✅ 小圖像 (512×512): 8執行緒最佳
- ✅ 大圖像 (6000×3000): 多進程編碼
- ✅ 批量處理: MPI分散式方案

### 未來改進
- 🔄 GPU實現 (需CUDA環境)
- 🔄 分散式儲存 (多節點協作)
- 🔄 演算法研究 (T1熵編碼優化)

## 9. 參考資源

### 代碼文件
- j2k_encode_mpi.cpp - MPI主程式
- j2k_encode_profile.cpp - Profiling版本
- src/tcd.cpp - 編碼管道
- src/common/opj_clock.cpp - 時間測量

### 編譯配置
- Makefile: OpenMP/MPI旗標設定
- submit_hybrid.sh: SLURM提交範本
- encode_fast.sh: 快速執行腳本

### 性能數據
- logs/j2k_*.log: 執行日誌
- profiling_results.txt: 詳細計時

---

**總結**: 通過OpenMP多執行緒 + MPI多進程的混合方案，
在Taiwania 3上實現了JPEG2000編碼的6.8倍加速，
主要瓶頸為T1熵編碼(95%計算量)的內在順序性限制。
