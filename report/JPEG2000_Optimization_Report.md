# JPEG2000 平行化加速 - 完整優化報告

###### tags: `平行程式設計` `JPEG2000` `OpenMP` `MPI` `HPC`

## 目錄
[TOC]

---

## 1. 專案概述

### 1.1 目標
優化 JPEG2000 影像壓縮/解壓縮的執行效率，在 Taiwania 3 HPC 上達到多核心加速。

### 1.2 實驗環境
| 項目 | 配置 |
|------|------|
| **平台** | Taiwania 3 HPC |
| **編譯器** | GCC 13.2.0 + OpenMPI 4.1.6 |
| **優化等級** | `-O3 -march=native -ffast-math` |
| **並行方案** | OpenMP + OpenJPEG thread_pool + MPI |

---

## 2. JPEG2000 編碼流程總覽

```
┌─────────────────────────────────────────────────────────────┐
│                  JPEG2000 Encoding Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Load Image (I/O)                                        │
│     └─ 讀取 PPM/PGM 檔案到記憶體                             │
│                                                             │
│  2. DC Level Shift                                          │
│     └─ 像素值減去 128 (去中心化)                            │
│                                                             │
│  3. MCT (Multi-Component Transform)                         │
│     └─ RGB → YCbCr 色彩轉換                                 │
│                                                             │
│  4. DWT (Discrete Wavelet Transform)                        │
│     └─ 多層小波分解                                          │
│                                                             │
│  5. T1 (Tier-1 Encoding) ⭐ 主要瓶頸                        │
│     └─ Codeblock 熵編碼 (MQC 算術編碼)                      │
│                                                             │
│  6. Rate Allocation                                         │
│     └─ R-D 最優位元分配                                      │
│                                                             │
│  7. T2 (Tier-2 Encoding)                                    │
│     └─ 位元流封裝                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 優化 1: Thread Pool / T1 並行化

### 3.1 位置
`src/t1.cpp` 第 2450-2515 行

### 3.2 原始問題
T1 編碼佔總時間 **95%**，是最大的瓶頸。

### 3.3 優化代碼

```cpp
OPJ_BOOL opj_t1_encode_cblks(opj_tcd_t* tcd,
                             opj_tcd_tile_t *tile,
                             opj_tcp_t *tcp,
                             const OPJ_FLOAT64 * mct_norms,
                             OPJ_UINT32 mct_numcomps)
{
    volatile OPJ_BOOL ret = OPJ_TRUE;
    opj_thread_pool_t* tp = tcd->thread_pool;  // ⬅️ 取得 thread pool
    OPJ_UINT32 compno, resno, bandno, precno, cblkno;
    opj_mutex_t* mutex = opj_mutex_create();

    tile->distotile = 0;

    // 遍歷所有 component → resolution → band → precinct → codeblock
    for (compno = 0; compno < tile->numcomps; ++compno) {
        opj_tcd_tilecomp_t* tilec = &tile->comps[compno];
        opj_tccp_t* tccp = &tcp->tccps[compno];

        for (resno = 0; resno < tilec->numresolutions; ++resno) {
            opj_tcd_resolution_t *res = &tilec->resolutions[resno];

            for (bandno = 0; bandno < res->numbands; ++bandno) {
                opj_tcd_band_t* OPJ_RESTRICT band = &res->bands[bandno];

                for (precno = 0; precno < res->pw * res->ph; ++precno) {
                    opj_tcd_precinct_t *prc = &band->precincts[precno];

                    for (cblkno = 0; cblkno < prc->cw * prc->ch; ++cblkno) {
                        opj_tcd_cblk_enc_t* cblk = &prc->cblks.enc[cblkno];

                        // ⬇️ 創建獨立的 job 結構
                        opj_t1_cblk_encode_processing_job_t* job =
                            (opj_t1_cblk_encode_processing_job_t*) opj_calloc(1,
                                    sizeof(opj_t1_cblk_encode_processing_job_t));
                        
                        job->compno = compno;
                        job->tile = tile;
                        job->cblk = cblk;
                        job->band = band;
                        job->tilec = tilec;
                        job->tccp = tccp;
                        job->mutex = mutex;
                        
                        // ⬇️ 提交到 thread pool 並行執行
                        opj_thread_pool_submit_job(tp, 
                            opj_t1_cblk_encode_processor,  // 處理函數
                            job);                          // 資料

                    } /* cblkno */
                } /* precno */
            } /* bandno */
        } /* resno  */
    } /* compno  */

    // ⬇️ 等待所有 job 完成
    opj_thread_pool_wait_completion(tcd->thread_pool, 0);
    
    if (mutex) {
        opj_mutex_destroy(mutex);
    }

    return ret;
}
```

### 3.4 為什麼可以並行？

```
Codeblock 結構:
┌─────────────────────────────────────────────┐
│ Tile                                        │
│  ├─ Component 0 (Y)                         │
│  │   ├─ Resolution 0                        │
│  │   │   ├─ Band LL                         │
│  │   │   │   ├─ Codeblock 0  ← Job 0       │
│  │   │   │   ├─ Codeblock 1  ← Job 1       │
│  │   │   │   └─ Codeblock 2  ← Job 2       │
│  │   │   └─ Band HL                         │
│  │   │       ├─ Codeblock 0  ← Job 3       │
│  │   │       └─ ...                         │
│  │   └─ Resolution 1                        │
│  │       └─ ...                             │
│  ├─ Component 1 (Cb)                        │
│  └─ Component 2 (Cr)                        │
└─────────────────────────────────────────────┘

✅ 每個 Codeblock 獨立編碼
✅ 無數據依賴 (不同 cblk 之間)
✅ 適合並行化
```

### 3.5 效果
| 配置 | T1 時間 | 加速比 |
|------|--------|--------|
| 1 core | 12.27s | 1.0x |
| 16 cores | 0.74s | **16.6x** |
| 32 cores | 0.43s | **28.5x** |

---

## 4. 優化 2: DWT 並行化

### 4.1 位置
`src/dwt.cpp` 第 2000-2100 行

### 4.2 優化代碼

```cpp
static INLINE OPJ_BOOL opj_dwt_encode_procedure(opj_thread_pool_t* tp,
        opj_tcd_tilecomp_t * tilec,
        void (*p_encode_and_deinterleave_v)(...),
        void (*p_encode_and_deinterleave_h_one_row)(...))
{
    const int num_threads = opj_thread_pool_get_thread_count(tp);
    
    // ... 初始化代碼 ...

    /* Vertical pass - 垂直方向濾波 */
    if (num_threads <= 1 || rw < 2 * NB_ELTS_V8) {
        // 單線程路徑
        for (j = 0; j + NB_ELTS_V8 <= rw; j += NB_ELTS_V8) {
            p_encode_and_deinterleave_v(tiledp + j, bj, rh, cas_col, w, NB_ELTS_V8);
        }
    } else {
        // ⬇️ 多線程路徑
        OPJ_UINT32 num_jobs = (OPJ_UINT32)num_threads;
        OPJ_UINT32 step_j = ((rw / num_jobs) / NB_ELTS_V8) * NB_ELTS_V8;

        for (j = 0; j < num_jobs; j++) {
            opj_dwt_encode_v_job_t* job;

            job = (opj_dwt_encode_v_job_t*) opj_malloc(sizeof(opj_dwt_encode_v_job_t));
            job->v.mem = (OPJ_INT32*)opj_aligned_32_malloc(l_data_size);
            job->v.dn = dn;
            job->v.sn = sn;
            job->tiledp = tiledp;
            job->min_j = j * step_j;
            job->max_j = (j + 1 == num_jobs) ? rw : (j + 1) * step_j;
            job->p_encode_and_deinterleave_v = p_encode_and_deinterleave_v;
            
            // ⬇️ 提交 job
            opj_thread_pool_submit_job(tp, opj_dwt_encode_v_func, job);
        }
        // ⬇️ 等待完成
        opj_thread_pool_wait_completion(tp, 0);
    }

    /* Horizontal pass - 水平方向濾波 */
    // ... 類似的並行化 ...
}
```

### 4.3 DWT 並行化原理

```
2D DWT 分解過程:

原始影像              第一層分解              第二層分解
┌────────────┐       ┌─────┬─────┐       ┌──┬──┬─────┐
│            │       │ LL  │ HL  │       │LL│HL│     │
│  影像數據   │  →    │─────┼─────│  →    │──┼──│ HL  │
│            │       │ LH  │ HH  │       │LH│HH│     │
│            │       │     │     │       ├──┴──┼─────┤
└────────────┘       └─────┴─────┘       │ LH  │ HH  │
                                         └─────┴─────┘

並行策略:
- 垂直方向: 不同列可並行處理
- 水平方向: 不同行可並行處理
```

### 4.4 效果
| 配置 | DWT 時間 | 加速比 |
|------|---------|--------|
| 1 core | 0.29s | 1.0x |
| 16 cores | 0.055s | **5.3x** |
| 32 cores | 0.034s | **8.5x** |

---

## 5. 優化 3: I/O Buffer 優化

### 5.1 位置
`j2k_encode_pnm.cpp` 第 74 行

### 5.2 優化代碼

```cpp
static opj_image_t* load_pnm_as_image(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open input: %s\n", path);
        return NULL;
    }

    // ⬇️ 設定 4MB 讀取緩衝區 (預設只有 4KB)
    (void)setvbuf(fp, NULL, _IOFBF, 4 * 1024 * 1024);

    // ... 讀取檔案 ...
}
```

### 5.3 為什麼有效？

```
檔案讀取模式比較:

小緩衝區 (4KB):
┌─────────────────────────────────────────────────┐
│ 系統呼叫: read() read() read() read() ...       │
│ 次數:     17000 次 (69MB ÷ 4KB)                 │
│ 開銷:     每次系統呼叫 ~1μs                       │
│ 總開銷:   ~17ms                                  │
└─────────────────────────────────────────────────┘

大緩衝區 (4MB):
┌─────────────────────────────────────────────────┐
│ 系統呼叫: read() read() read() ...              │
│ 次數:     17 次 (69MB ÷ 4MB)                    │
│ 開銷:     每次 ~1μs                              │
│ 總開銷:   ~0.017ms                               │
└─────────────────────────────────────────────────┘
```

### 5.4 效果
| Buffer Size | Load 時間 | 加速 |
|-------------|----------|------|
| 4KB | 0.177s | 1.0x |
| 4MB | 0.143s | **1.24x** |

---

## 6. 優化 4: Pixel Copy 並行化

### 6.1 位置
`j2k_decode_pnm.cpp` 第 87-100 行

### 6.2 優化代碼

```cpp
static int write_pnm_u8(const char* path, const opj_image_t* image) {
    // ... 初始化 ...
    
    if (numcomps == 3) {
        unsigned char* buf = (unsigned char*)malloc(pixels * 3);
        
        const int rshift = (int)image->comps[0].prec - 8;
        const int gshift = (int)image->comps[1].prec - 8;
        const int bshift = (int)image->comps[2].prec - 8;

        // ⬇️ OpenMP 並行化像素轉換
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < pixels; i++) {
            int r = image->comps[0].data[i];  // 讀取 R 通道
            int g = image->comps[1].data[i];  // 讀取 G 通道
            int b = image->comps[2].data[i];  // 讀取 B 通道

            // 位元精度調整
            if (rshift > 0) r >>= rshift;
            if (gshift > 0) g >>= gshift;
            if (bshift > 0) b >>= bshift;

            // 寫入交錯格式
            buf[i * 3 + 0] = clamp_u8(r);
            buf[i * 3 + 1] = clamp_u8(g);
            buf[i * 3 + 2] = clamp_u8(b);
        }

        // 一次性寫入檔案
        fwrite(buf, 1, pixels * 3, fp);
        free(buf);
    }
    // ...
}
```

### 6.3 為什麼可以並行？

```
記憶體存取模式:

原始數據（分離通道）:
comps[0].data: [R0, R1, R2, R3, R4, R5, R6, R7, ...]
comps[1].data: [G0, G1, G2, G3, G4, G5, G6, G7, ...]
comps[2].data: [B0, B1, B2, B3, B4, B5, B6, B7, ...]

輸出（交錯格式）:
buf: [R0, G0, B0, R1, G1, B1, R2, G2, B2, ...]

並行分配 (4 threads, 8 pixels):
Thread 0: 處理像素 0-1 → buf[0:5]
Thread 1: 處理像素 2-3 → buf[6:11]
Thread 2: 處理像素 4-5 → buf[12:17]
Thread 3: 處理像素 6-7 → buf[18:23]

✅ 每個像素完全獨立
✅ 無數據依賴
✅ 記憶體存取連續 → cache friendly
```

### 6.4 效果
| 配置 | Write 時間 | 加速比 |
|------|-----------|--------|
| 1 core | ~200ms | 1.0x |
| 16 cores | ~50ms | **4x** |

---

## 7. 優化 5: MPI Tile 分割

### 7.1 位置
`j2k_encode_mpi.cpp` (編譯時加 `-DTILE_PARALLEL`)

### 7.2 整體流程圖

```
┌─────────────────────────────────────────────────────────────────┐
│                    MPI Tile Parallel 流程                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: MPI 初始化                                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MPI_Init()                                                │   │
│  │ MPI_Comm_rank() → 取得自己的編號 (0, 1, 2, ...)          │   │
│  │ MPI_Comm_size() → 取得總共幾個 processes                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  Step 2: Rank 0 讀取完整圖片                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ if (rank == 0) {                                          │   │
│  │     full_image = load_pnm_as_image(input.ppm);            │   │
│  │ }                                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  Step 3: 廣播圖片資訊給所有 ranks                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MPI_Bcast(&width, ...)   ← 所有 rank 得到相同的 width    │   │
│  │ MPI_Bcast(&height, ...)  ← 所有 rank 得到相同的 height   │   │
│  │ MPI_Bcast(&numcomps, ...)                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  Step 4: 計算每個 rank 負責的區域                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ tile_height = height / size;                              │   │
│  │ my_start_row = rank * tile_height + ...                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  Step 5: Rank 0 分發數據 (MPI_Send/Recv)                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Rank 0: MPI_Send() 發送 tile 給 Rank 1, 2, 3...           │   │
│  │ Rank 1~N: MPI_Recv() 接收屬於自己的 tile                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  Step 6: 每個 rank 獨立編碼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ encode_tile(tile_image, output_tile{rank}.j2k)            │   │
│  │ (內部使用 OpenMP thread_pool 並行化 T1/DWT)               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  Step 7: 結束                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MPI_Finalize()                                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Step 1: MPI 初始化

```cpp
int main(int argc, char** argv) {
    int rank, size;
    
    // ⬇️ 初始化 MPI 環境
    MPI_Init(&argc, &argv);
    
    // ⬇️ 取得自己的編號 (0, 1, 2, ...)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // ⬇️ 取得總共有幾個 MPI processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 例如: srun -n 4 執行時
    // Rank 0: rank=0, size=4
    // Rank 1: rank=1, size=4
    // Rank 2: rank=2, size=4
    // Rank 3: rank=3, size=4
```

### 7.4 Step 2-3: 讀取並廣播圖片資訊

```cpp
    opj_image_t* full_image = NULL;
    int width = 0, height = 0, numcomps = 0;

    // ⬇️ 只有 Rank 0 讀取完整圖片
    if (rank == 0) {
        profile_times_t dummy_prof = {0};
        full_image = load_pnm_as_image(in_path, &dummy_prof);
        
        width = full_image->x1 - full_image->x0;    // 6016
        height = full_image->y1 - full_image->y0;   // 4000
        numcomps = full_image->numcomps;            // 3 (RGB)
        
        printf("[Rank 0] Loaded image: %dx%d\n", width, height);
    }

    // ⬇️ 廣播圖片尺寸給所有 ranks
    // MPI_Bcast: 從 root (rank 0) 發送到所有其他 ranks
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numcomps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 執行後，所有 ranks 都知道:
    // width = 6016, height = 4000, numcomps = 3
```

```
MPI_Bcast 示意圖:

執行前:
  Rank 0: width=6016  ← 有值
  Rank 1: width=0     ← 沒值
  Rank 2: width=0     ← 沒值
  Rank 3: width=0     ← 沒值

MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
              │      │    │    │
              │      │    │    └─ 從 rank 0 廣播
              │      │    └─ 資料型別
              │      └─ 1 個元素
              └─ 變數位址

執行後:
  Rank 0: width=6016  ✓
  Rank 1: width=6016  ✓
  Rank 2: width=6016  ✓
  Rank 3: width=6016  ✓
```

### 7.5 Step 4: 計算每個 Rank 負責的區域

```cpp
    // ⬇️ 計算每個 rank 負責多少行
    int tile_height = height / size;           // 4000 / 4 = 1000
    int remainder = height % size;             // 4000 % 4 = 0
    
    // ⬇️ 處理不能整除的情況 (前幾個 rank 多分一行)
    int my_tile_height = tile_height + (rank < remainder ? 1 : 0);
    
    // ⬇️ 計算自己負責的起始行
    int my_start_row = rank * tile_height + (rank < remainder ? rank : remainder);

    printf("[Rank %d] Tile: rows %d to %d (height=%d)\n", 
           rank, my_start_row, my_start_row + my_tile_height - 1, my_tile_height);
```

```
分割範例 (height=4000, size=4):

tile_height = 4000 / 4 = 1000
remainder   = 4000 % 4 = 0 (剛好整除)

Rank 0:
  my_tile_height = 1000
  my_start_row   = 0
  負責: rows 0 ~ 999

Rank 1:
  my_tile_height = 1000
  my_start_row   = 1000
  負責: rows 1000 ~ 1999

Rank 2:
  my_tile_height = 1000
  my_start_row   = 2000
  負責: rows 2000 ~ 2999

Rank 3:
  my_tile_height = 1000
  my_start_row   = 3000
  負責: rows 3000 ~ 3999
```

```
不能整除的範例 (height=4002, size=4):

tile_height = 4002 / 4 = 1000
remainder   = 4002 % 4 = 2

Rank 0: (rank < 2) → +1
  my_tile_height = 1001
  my_start_row   = 0
  負責: rows 0 ~ 1000

Rank 1: (rank < 2) → +1
  my_tile_height = 1001
  my_start_row   = 1001
  負責: rows 1001 ~ 2001

Rank 2: (rank >= 2) → +0
  my_tile_height = 1000
  my_start_row   = 2002
  負責: rows 2002 ~ 3001

Rank 3: (rank >= 2) → +0
  my_tile_height = 1000
  my_start_row   = 3002
  負責: rows 3002 ~ 4001
```

### 7.6 Step 5: 分發數據

```cpp
    // ⬇️ 創建 tile 圖片結構 (每個 rank 都執行)
    opj_image_cmptparm_t cmptparms[3];
    for (int i = 0; i < numcomps; i++) {
        cmptparms[i].w = width;
        cmptparms[i].h = my_tile_height;  // ← 只有自己負責的高度
        cmptparms[i].prec = 8;
        // ...
    }
    opj_image_t* tile_image = opj_image_create(numcomps, cmptparms, OPJ_CLRSPC_SRGB);

    int pixels_per_tile = width * my_tile_height;  // 6016 * 1000

    // ⬇️ 數據分發
    if (rank == 0) {
        // ========== Rank 0 的工作 ==========
        
        // 1. 複製自己的 tile (rows 0 ~ 999)
        for (int c = 0; c < numcomps; c++) {
            for (int row = 0; row < my_tile_height; row++) {
                memcpy(&tile_image->comps[c].data[row * width],
                       &full_image->comps[c].data[row * width],
                       width * sizeof(OPJ_INT32));
            }
        }
        
        // 2. 發送 tile 給其他 ranks
        for (int r = 1; r < size; r++) {
            // 計算 rank r 的起始行和 tile 高度
            int r_tile_height = tile_height + (r < remainder ? 1 : 0);
            int r_start_row = r * tile_height + (r < remainder ? r : remainder);
            int r_pixels = width * r_tile_height;
            
            // 發送每個 color component
            for (int c = 0; c < numcomps; c++) {
                MPI_Send(&full_image->comps[c].data[r_start_row * width],
                         r_pixels,       // 發送多少個 int
                         MPI_INT,        // 資料型別
                         r,              // 目標 rank
                         c,              // tag (用 component index)
                         MPI_COMM_WORLD);
            }
        }
        
        // 3. 釋放完整圖片 (不再需要)
        opj_image_destroy(full_image);
        
    } else {
        // ========== 其他 Ranks 的工作 ==========
        
        // 接收屬於自己的 tile
        for (int c = 0; c < numcomps; c++) {
            MPI_Recv(tile_image->comps[c].data,
                     pixels_per_tile,     // 接收多少個 int
                     MPI_INT,             // 資料型別
                     0,                   // 從 rank 0 接收
                     c,                   // tag (對應 component)
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    }
```

```
MPI_Send / MPI_Recv 示意圖:

原始圖片 (只在 Rank 0 的記憶體):
┌──────────────────────────────────────────────────────┐
│ comps[0].data (R): [pixel0, pixel1, ..., pixel24M]   │
│ comps[1].data (G): [pixel0, pixel1, ..., pixel24M]   │
│ comps[2].data (B): [pixel0, pixel1, ..., pixel24M]   │
└──────────────────────────────────────────────────────┘

分發過程:

Rank 0                    Rank 1                    Rank 2
┌─────────┐              ┌─────────┐              ┌─────────┐
│ Rows    │              │         │              │         │
│ 0-999   │ ← memcpy     │         │              │         │
│ (local) │              │         │              │         │
├─────────┤              │         │              │         │
│ Rows    │─── Send ────→│ Rows    │              │         │
│ 1000-   │              │ 1000-   │ ← Recv       │         │
│ 1999    │              │ 1999    │              │         │
├─────────┤              └─────────┘              │         │
│ Rows    │─────────────── Send ─────────────────→│ Rows    │
│ 2000-   │                                       │ 2000-   │
│ 2999    │                                       │ 2999    │
├─────────┤                                       └─────────┘
│ Rows    │
│ 3000-   │─── Send ────→ Rank 3
│ 3999    │
└─────────┘
```

### 7.7 Step 6: 獨立編碼

```cpp
    // ⬇️ 每個 rank 產生自己的輸出檔名
    char out_path[512];
    snprintf(out_path, sizeof(out_path), "%s_tile%d.j2k", out_prefix, rank);
    // 例如: output_tile0.j2k, output_tile1.j2k, ...

    // ⬇️ 編碼自己的 tile (使用 OpenMP 並行化)
    int result = encode_tile_image(tile_image, out_path, rank);
    
    // encode_tile_image 內部:
    // - 使用 opj_codec_set_threads() 設定 OpenMP 線程數
    // - T1 編碼用 thread_pool 並行化
    // - DWT 編碼用 thread_pool 並行化
```

```
並行編碼示意圖:

時間 →
────────────────────────────────────────────────────────────

Rank 0: [======= 編碼 Tile 0 =======]
         T1: 16 threads
         DWT: 16 threads

Rank 1: [======= 編碼 Tile 1 =======]
         T1: 16 threads
         DWT: 16 threads

Rank 2: [======= 編碼 Tile 2 =======]
         T1: 16 threads
         DWT: 16 threads

Rank 3: [======= 編碼 Tile 3 =======]
         T1: 16 threads
         DWT: 16 threads

────────────────────────────────────────────────────────────

✅ 4 個 MPI ranks 同時編碼
✅ 每個 rank 內部用 16 個 OpenMP threads
✅ 總共 4 × 16 = 64 個並行執行單元
```

### 7.8 MPI 分割原理

```
原始圖片 (6016 × 4000):
┌─────────────────────────────┐
│                             │
│         Full Image          │
│                             │
│                             │
└─────────────────────────────┘

MPI 分割 (4 ranks):
┌─────────────────────────────┐
│  Rank 0: rows 0-999         │ ← Tile 0 (1000 行) → tile0.j2k
├─────────────────────────────┤
│  Rank 1: rows 1000-1999     │ ← Tile 1 (1000 行) → tile1.j2k
├─────────────────────────────┤
│  Rank 2: rows 2000-2999     │ ← Tile 2 (1000 行) → tile2.j2k
├─────────────────────────────┤
│  Rank 3: rows 3000-3999     │ ← Tile 3 (1000 行) → tile3.j2k
└─────────────────────────────┘

✅ 每個 rank 獨立編碼 → 真正的並行
✅ 可跨節點擴展
✅ 輸出多個 .j2k 檔案 (可用 merge_tiles 合併)
```

### 7.9 Single Process vs Multi-Process

#### Single Process 時：

```cpp
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // rank = 0
MPI_Comm_size(MPI_COMM_WORLD, &size);  // size = 1 ← 只有 1 個 process!

// 計算 tile
int tile_height = height / size;       // 4000 / 1 = 4000 (整個圖片)
int remainder = height % size;         // 4000 % 1 = 0

int my_tile_height = tile_height + (0 < 0 ? 1 : 0);  // 4000
int my_start_row = 0 * 4000 + 0;                      // 0

// 只有 rank 0 執行
if (rank == 0) {
    // ✅ 讀完整圖片
    full_image = load_pnm_as_image(...);
    
    // ✅ 複製自己的 tile (實際上是整個圖片)
    for (int c = 0; c < numcomps; c++) {
        for (int row = 0; row < 4000; row++) {  // 迴圈整個高度
            memcpy(...);
        }
    }
    
    // ❌ 沒有其他 rank 要發送
    for (int r = 1; r < 1; r++) {  // 迴圈不執行 (1 < 1 = false)
        // 這段不會執行
    }
} else {
    // ❌ 這段永遠不會執行 (rank != 0)
}

// ✅ 編碼整個圖片
encode_tile_image(tile_image, "output_tile0.j2k", 0);
```

```
流程圖 (Single Process):

┌────────────────────────────────────────┐
│ srun -n 1 (只有 1 個 process)          │
│ rank = 0, size = 1                    │
└────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────┐
│ Rank 0: 讀取完整圖片 (4000 行)          │
│         [====== Full Image ======]    │
└────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────┐
│ MPI_Bcast: 廣播給 size-1=0 個 rank    │
│ (沒有其他 rank，所以沒作用)            │
└────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────┐
│ Rank 0: MPI_Send 迴圈不執行            │
│ (for (int r=1; r < 1; r++) 不成立)    │
└────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────┐
│ Rank 0: 編碼整個圖片                    │
│ encode_tile_image(full_image, ...)     │
└────────────────────────────────────────┘
```

#### Multi-Process 時 (正常情況)：

```cpp
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // rank = 0, 1, 2, 3
MPI_Comm_size(MPI_COMM_WORLD, &size);  // size = 4 ← 4 個 processes!

// 計算 tile
int tile_height = height / size;       // 4000 / 4 = 1000
int remainder = height % size;         // 4000 % 4 = 0

// Rank 0: tile_height = 1000, my_start_row = 0 (rows 0-999)
// Rank 1: tile_height = 1000, my_start_row = 1000 (rows 1000-1999)
// Rank 2: tile_height = 1000, my_start_row = 2000 (rows 2000-2999)
// Rank 3: tile_height = 1000, my_start_row = 3000 (rows 3000-3999)

if (rank == 0) {
    // ✅ 只有 Rank 0 讀完整圖片
    full_image = load_pnm_as_image(...);
    
    // ✅ 複製自己的 tile
    for (int c = 0; c < numcomps; c++) {
        for (int row = 0; row < 1000; row++) {  // 只迴圈 1000 行
            memcpy(&tile_image->comps[c].data[row * width],
                   &full_image->comps[c].data[row * width], ...);
        }
    }
    
    // ✅ 發送 tile 給其他 ranks
    for (int r = 1; r < 4; r++) {  // r = 1, 2, 3
        MPI_Send(..., r, ...);
    }
} else {
    // ✅ Rank 1, 2, 3 接收自己的 tile
    for (int c = 0; c < numcomps; c++) {
        MPI_Recv(tile_image->comps[c].data, pixels_per_tile, ...);
    }
}

// ✅ 每個 rank 獨立編碼自己的 tile
// Rank 0: encode_tile_image(tile0, "output_tile0.j2k", 0);
// Rank 1: encode_tile_image(tile1, "output_tile1.j2k", 1);
// Rank 2: encode_tile_image(tile2, "output_tile2.j2k", 2);
// Rank 3: encode_tile_image(tile3, "output_tile3.j2k", 3);
```

```
流程圖 (Multi-Process):

┌──────────────────────────────────────────┐
│ srun -n 4 (4 個 processes)               │
│ rank = 0, 1, 2, 3; size = 4             │
└──────────────────────────────────────────┘
         ↓         ↓         ↓         ↓
┌────────┴────┐ ┌─┴───────┐ ┌─┴───────┐ ┌─┴───────┐
│ Rank 0      │ │ Rank 1  │ │ Rank 2  │ │ Rank 3  │
│ 讀全圖      │ │ 等待    │ │ 等待    │ │ 等待    │
└────┬────────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │                │           │           │
     └────────────────┴───────────┴───────────→ MPI_Bcast
                                                (廣播尺寸)
     ↓                ↓           ↓           ↓
┌────────────┐  ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Rank 0     │  │ Rank 1  │ │ Rank 2  │ │ Rank 3  │
│ 複製 rows  │  │ 等待接收│ │ 等待接收│ │ 等待接收│
│ 0-999      │  │         │ │         │ │         │
└────┬───────┘  └────┬────┘ └────┬────┘ └────┬────┘
     │                │           │           │
     ├─ MPI_Send ────→│           │           │
     │                │           │           │
     └─ MPI_Send ─────────────────→│           │
     │                           │           │
     └─ MPI_Send ───────────────────────────→│
     ↓                ↓           ↓           ↓
  編碼           編碼            編碼            編碼
  tile0.j2k    tile1.j2k        tile2.j2k      tile3.j2k
  (1000 行)    (1000 行)        (1000 行)      (1000 行)
```

### 7.10 何時使用 MPI Tile 分割？

```
+─────────────────────────────────────────────────────────+
│ 選擇 Single Process vs Multi-Process MPI              │
+─────────────────────────────────────────────────────────+
│                                                         │
│ 使用 Single Process (-DTILE_PARALLEL 不編譯)          │
│ ├─ srun -n 1 ./j2k_encode_pnm                         │
│ ├─ 適合: 單個大圖片, 單節點多核心                       │
│ ├─ 優點: 簡單, 無通訊開銷                               │
│ └─ 缺點: 無法跨節點                                     │
│                                                         │
│ 使用 Multi-Process (-DTILE_PARALLEL 編譯)             │
│ ├─ srun -n 4 ./j2k_encode_mpi_tile                    │
│ ├─ 適合: 非常大的圖片, 需要跨節點                       │
│ ├─ 優點: 真正的並行, 可跨節點擴展                       │
│ └─ 缺點: 通訊開銷, 需要 MPI 環境                        │
│                                                         │
+─────────────────────────────────────────────────────────+
```

### 7.11 效果
| 配置 | 時間 | 加速比 |
|------|------|--------|
| 1N 1n 32c (OpenMP only) | 0.85s | 1.0x |
| 1N 2n 16c (MPI + OpenMP) | 0.80s | **1.06x** |
| 1N 4n 8c (MPI + OpenMP) | 0.70s | **1.21x** |

---

## 8. Thread Pool 設定

### 8.1 位置
`j2k_encode_pnm.cpp` 第 242-252 行

### 8.2 代碼

```cpp
int main(int argc, char** argv) {
    // ... 初始化 ...
    
    opj_codec_t* codec = opj_create_compress(OPJ_CODEC_J2K);
    
    // ⬇️ 從環境變數讀取線程數
    int num_threads = 4;  // 預設 4 線程
    const char* env_threads = getenv("OMP_NUM_THREADS");
    if (env_threads) {
        num_threads = atoi(env_threads);
    }
    
    // ⬇️ 設定 OpenJPEG thread pool
    if (num_threads > 0) {
        if (!opj_codec_set_threads(codec, num_threads)) {
            fprintf(stderr, "Warning: Failed to set %d threads\n", num_threads);
        } else {
            fprintf(stdout, "Using %d threads for encoding\n", num_threads);
        }
    }
    
    // ... 編碼 ...
}
```

### 8.3 Thread Pool 內部運作

```cpp
// src/j2k.cpp

OPJ_BOOL opj_j2k_set_threads(opj_j2k_t *j2k, OPJ_UINT32 num_threads)
{
    // 銷毀舊的 thread pool
    opj_thread_pool_destroy(j2k->m_tp);
    
    // ⬇️ 創建新的 thread pool
    j2k->m_tp = opj_thread_pool_create((int)num_threads);
    
    return (j2k->m_tp != NULL);
}
```

```
Thread Pool 架構:
┌─────────────────────────────────────────────────┐
│ opj_thread_pool_t                               │
├─────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Thread 0│  │ Thread 1│  │Thread N │         │
│  └────┬────┘  └────┬────┘  └────┬────┘         │
│       │            │            │               │
│       v            v            v               │
│  ┌─────────────────────────────────────────┐   │
│  │           Job Queue                      │   │
│  │  [Job0] [Job1] [Job2] [Job3] ...         │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  opj_thread_pool_submit_job(tp, fn, data)      │
│  opj_thread_pool_wait_completion(tp, 0)        │
└─────────────────────────────────────────────────┘
```

---

## 9. Ablation Study (消融實驗)

### 9.1 測試圖片
`dataset/02.ppm` (6016×4000, 69 MB)

### 9.2 結果

| Profile Time | DWT | I/O (Load) | T1 | Total | 加速比 |
|--------------|-----|------------|----|----|--------|
| **baseline (1 core)** | 0.29s | 0.17s | 12.27s | **12.90s** | 1.0x |
| **thread_pool (16 cores)** | 0.055s | 0.12s | 0.74s | **1.10s** | **11.7x** |
| **+ IO buffer (16 cores)** | 0.055s | 0.10s | 0.74s | **1.05s** | 12.3x |
| **Full opt (32 cores)** | 0.034s | 0.23s | 0.43s | **0.85s** | **15.2x** |

### 9.3 各優化貢獻

```
加速貢獻分解:

baseline: 12.90s
    │
    ├─ T1 thread_pool: -11.53s (89.4%)  ⭐ 最大貢獻
    │
    ├─ DWT thread_pool: -0.24s (1.8%)
    │
    ├─ IO buffer: -0.05s (0.4%)
    │
    ├─ 32 cores vs 16: -0.25s (1.9%)
    │
    └─ Final: 0.85s (15.2x 加速)
```

---

## 10. 執行方式

### 10.1 編譯

```bash
# 載入模組
module load gcc/13.2.0 openmpi/4.1.6

# 編譯所有版本
make clean && make -j8
```

### 10.2 單節點 OpenMP

```bash
# 設定線程數
export OMP_NUM_THREADS=32

# 執行
srun --account=ACD114118 -N 1 -n 1 -c 32 \
    ./build/j2k_encode_pnm input.ppm output.j2k
```

### 10.3 MPI Tile 模式

```bash
# 設定環境
export OMP_NUM_THREADS=16
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_coll=^hcoll

# 執行 (2 MPI ranks × 16 threads = 32 cores)
srun --account=ACD114118 -N 1 -n 2 -c 16 \
    ./build/j2k_encode_mpi_tile input.ppm output_prefix
```

---

## 11. 結論

### 11.1 成果
- ✅ **15.2x 加速** (單節點 32 cores)
- ✅ **16.1x 加速** (MPI tile 模式)
- ✅ T1 編碼加速 **28.5x**
- ✅ DWT 加速 **8.5x**

### 11.2 關鍵優化

| 排名 | 優化 | 貢獻 |
|------|------|------|
| 1 | T1 thread_pool | 89.4% |
| 2 | DWT thread_pool | 1.8% |
| 3 | 增加核心數 | 1.9% |
| 4 | IO buffer | 0.4% |

### 11.3 限制
- ❌ T1 內部有順序依賴，無法達到線性加速
- ❌ MPI 跨節點通訊開銷較大
- ❌ 小圖片並行化開銷大於收益

---

## 12. 參考資料

- [OpenJPEG 官方文檔](https://github.com/uclouvain/openjpeg)
- [JPEG2000 標準 (ISO/IEC 15444-1)](https://www.iso.org/standard/78321.html)
- [OpenMP 規範](https://www.openmp.org/specifications/)
- [MPI 標準](https://www.mpi-forum.org/docs/)
