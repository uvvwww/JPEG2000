# Minimal OpenJPEG J2K Project (C++ Version)

這是一個精簡版的 OpenJPEG 專案，專注於 J2K 核心，已轉換為 C++ 以支援 OpenMP 等平行化開發。

## 目錄結構

*   `src/`: OpenJPEG 的核心原始碼 (.cpp, .h)。**這是你要修改來做平行化的地方。**
*   `j2k_encode_pnm.cpp`: 極簡的編碼器主程式 (讀取 PGM/PPM -> 輸出 J2K)。
*   `j2k_decode_pnm.cpp`: 極簡的解碼器主程式 (讀取 J2K -> 輸出 PGM/PPM)。
*   `Makefile`: 建置腳本 (使用 g++)。
*   `build/`: 編譯後的執行檔與靜態庫存放處。

## 如何編譯 (Build)

在終端機 (WSL/Linux) 進入此目錄並執行 `make`：

```bash
cd j2k_only
make -j
```

這會產生兩個執行檔：
1.  `build/j2k_encode_pnm`
2.  `build/j2k_decode_pnm`

## 如何使用 (Usage)

### 1. 編碼 (壓縮)

輸入圖片必須是 **PGM (灰階)** 或 **PPM (彩色)** 格式 (P5/P6 binary)。

```bash
# 用法: ./build/j2k_encode_pnm <input.ppm> <output.j2k>
./build/j2k_encode_pnm input.ppm output.j2k
```

### 2. 解碼 (解壓縮)

```bash
# 用法: ./build/j2k_decode_pnm <input.j2k> <output.ppm>
./build/j2k_decode_pnm output.j2k restored.ppm
```

## 什麼是 PPM 格式? (What is PPM?)

**PPM (Portable Pixel Map)** 是一種極其簡單的無壓縮圖片格式，常在學術與開發中作為中間格式使用。

*   **結構簡單**：它只有一個純文字檔頭 (Header) 描述寬、高、最大顏色值，接著就是原始的 RGB 像素數據 (Binary)。
*   **無壓縮**：因為沒有壓縮，讀寫程式碼非常容易撰寫 (只需幾行 C 語言)，這也是為什麼這個極簡專案選擇支援它的原因。
*   **檔案大**：因為沒有壓縮，檔案體積通常很大 (例如一張 1920x1080 的圖片約需 6MB)。
*   **PGM**：是 PPM 的灰階版本 (Portable Gray Map)。

## 圖片格式轉換 (Image Conversion)

由於此極簡程式只支援 PGM/PPM 格式，建議使用 **ImageMagick** 來轉換你的 JPG/PNG 圖片。

### 安裝 ImageMagick (WSL/Ubuntu)

```bash
sudo apt update
sudo apt install imagemagick
```

### 使用方法

**1. 將 JPG/PNG 轉為 PPM (給編碼器用):**

```bash
# 將 input.jpg 轉為 input.ppm
convert input.jpg input.ppm
```

**2. 將 PPM 轉回 JPG/PNG (檢視解碼結果):**

```bash
# 將 restored.ppm 轉為 restored.jpg
convert restored.ppm restored.jpg
```

## 開發提示

*   如果你要進行平行化 (例如使用 Pthreads 或 OpenMP)，請直接修改 `src/` 資料夾內的 `.c` 檔案 (例如 `t1.c`, `t2.c`, `dwt.c` 等)。
*   修改後，只需重新執行 `make` 即可重新編譯。
*   若要清除編譯檔案，請執行 `make clean`。

## 專案來源 (Origin)

本專案程式碼衍生自 OpenJPEG 官方儲存庫：
<https://github.com/uclouvain/openjpeg>

## 程式運作流程 (Workflow Description)

本專案的主程式 (`j2k_encode_pnm.cpp` 與 `j2k_decode_pnm.cpp`) 如何呼叫 OpenJPEG 核心函式庫的流程說明：

### 編碼流程 (Encoder Workflow)

1.  **讀取輸入 (Read Input)**: 主程式讀取 PPM/PGM 檔案，將 RGB 像素數據載入記憶體。
2.  **轉換格式 (Convert to opj_image_t)**: 將原始像素數據轉換為 OpenJPEG 內部的 `opj_image_t` 結構。
3.  **設定參數 (Setup Parameters)**: 初始化 `opj_cparameters_t`，設定壓縮參數 (如無失真壓縮、Tile 大小等)。
4.  **建立壓縮器 (Create Compressor)**: 呼叫 `opj_create_compress(OPJ_CODEC_J2K)` 建立 J2K 壓縮物件。
5.  **設定事件 (Setup Events)**: 設定錯誤與訊息輸出的 Callback 函式。
6.  **初始化編碼器 (Init Encoder)**: 呼叫 `opj_setup_encoder`。
7.  **建立串流 (Create Stream)**: 呼叫 `opj_stream_create_default_file_stream` 開啟輸出檔案串流。
8.  **開始壓縮 (Start Compress)**: 呼叫 `opj_start_compress` 寫入檔頭。
9.  **執行編碼 (Encode)**: 呼叫 `opj_encode` 進行主要的壓縮運算 (這是最耗時的步驟)。
    *   **這是平行化的主要目標**。
    *   `opj_encode` (在 `src/j2k.cpp`) 會呼叫 `tcd_encode_tile` (在 `src/tcd.cpp`)。
    *   `tcd_encode_tile` 會依序呼叫以下核心演算法 (資料流方向：影像 -> 壓縮檔)：
        1.  **`mct_encode` (在 `src/mct.cpp`)**: Multi-Component Transform。
            *   **功能**: 進行色彩空間轉換 (例如 RGB -> YUV)。
            *   **關係**: 必須在 DWT 之前執行，目的是去除各個顏色通道 (Components) 之間的相關性，提高壓縮效率。
        2.  **`dwt_encode` (在 `src/dwt.cpp`)**: Discrete Wavelet Transform。
            *   **功能**: 將影像分解為不同頻率的子頻帶 (Subbands)。
            *   **關係**: 接收 MCT 轉換後的數據進行處理。
        3.  **`t1_encode_cblks` (在 `src/t1.cpp`)**: Tier-1 Entropy Coding。
            *   **功能**: 對 DWT 係數進行算術編碼。
            *   **關係**: **這是最耗時的部分 (約佔 70%+)**，也是 OpenMP 平行化的首要目標。
        4.  **`t2_encode_packets` (在 `src/t2.cpp`)**: Tier-2 Coding (Packetization)。
            *   **功能**: 將編碼後的數據打包成 J2K 格式的 Packet。
10. **結束壓縮 (End Compress)**: 呼叫 `opj_end_compress` 寫入結尾標記。
11. **清理 (Cleanup)**: 釋放記憶體與關閉檔案。

### 解碼流程 (Decoder Workflow)

1.  **建立串流 (Create Stream)**: 開啟 J2K 輸入檔案串流。
2.  **建立解壓縮器 (Create Decompressor)**: 呼叫 `opj_create_decompress(OPJ_CODEC_J2K)`。
3.  **設定參數 (Setup Parameters)**: 初始化 `opj_dparameters_t` 並呼叫 `opj_setup_decoder`。
4.  **讀取檔頭 (Read Header)**: 呼叫 `opj_read_header` 讀取圖片尺寸與參數，建立 `opj_image_t` 結構。
5.  **執行解碼 (Decode)**: 呼叫 `opj_decode` 進行解壓縮運算。
    *   **這是平行化的主要目標**。
    *   `opj_decode` (在 `src/j2k.cpp`) 會呼叫 `tcd_decode_tile` (在 `src/tcd.cpp`)。
    *   `tcd_decode_tile` 會依序呼叫以下核心演算法 (資料流方向：壓縮檔 -> 影像)：
        1.  **`t2_decode_packets` (在 `src/t2.cpp`)**: Tier-2 Decoding。
            *   **功能**: 讀取 Packet 標頭與數據。
        2.  **`t1_decode_cblks` (在 `src/t1.cpp`)**: Tier-1 Entropy Decoding。
            *   **功能**: 將壓縮數據還原為 DWT 係數。
            *   **關係**: **這是最耗時的部分**。
        3.  **`dwt_decode` (在 `src/dwt.cpp`)**: Discrete Wavelet Transform (Inverse)。
            *   **功能**: 將頻率域係數還原為空間域影像數據。
        4.  **`mct_decode` (在 `src/mct.cpp`)**: Multi-Component Transform (Inverse)。
            *   **功能**: 進行色彩空間反轉換 (例如 YUV -> RGB)。
            *   **關係**: 必須在 DWT 之後執行，將各個獨立的通道還原為最終的彩色影像。
6.  **結束解碼 (End Decompress)**: 呼叫 `opj_end_decompress`。
7.  **輸出圖片 (Write Output)**: 將解碼後的 `opj_image_t` 數據轉換回 RGB 格式並寫入 PPM/PGM 檔案。
8.  **清理 (Cleanup)**: 釋放資源。

## 原始碼檔案說明 (Source Files Description)

`src/` 目錄下的檔案用途說明 (已移除不必要的 JPIP 與 JP2 檔案)：

*   **核心編解碼流程**
    *   `openjpeg.cpp`: OpenJPEG 公開 API 介面實作 (進入點)。
    *   `j2k.cpp`: JPEG 2000 Codestream 解析與寫入的主要邏輯。
    *   `tcd.cpp`: **Tile Coder/Decoder**。負責調度整個 Tile 的編解碼流程 (T1, T2, DWT, MCT)。
    *   `image.cpp`: 影像資料結構 (`opj_image_t`) 的建立與管理。

*   **編碼演算法 (平行化重點)**
    *   `t1.cpp`: **Tier-1 Coding** (EBCOT)。負責算術編碼 (Entropy Coding)，是運算量最大的部分。
    *   `dwt.cpp`: **Discrete Wavelet Transform** (離散小波轉換)。負責影像頻率分解。
    *   `mct.cpp`: **Multi-Component Transform**。負責色彩空間轉換 (如 RGB <-> YUV)。
    *   `t2.cpp`: **Tier-2 Coding**。負責將壓縮後的數據打包成 Packet。
    *   `mqc.cpp`: MQ Coder。底層的算術編碼器實作。

*   **I/O 與 工具**
    *   `bio.cpp`: Bit Input/Output。位元級別的讀寫操作。
    *   `cio.cpp`: Byte Input/Output。串流 (Stream) 級別的讀寫操作。
    *   `event.cpp`: 事件管理 (錯誤、警告、訊息 callback)。
    *   `opj_clock.cpp`: 計時工具。
    *   `opj_malloc.cpp`: 記憶體配置包裝。
    *   `pi.cpp`: Packet Iterator。負責決定 Packet 的順序 (Progression Order)。
    *   `tgt.cpp`: Tag Tree Coder。用於管理 Codeblock 的包含資訊。
    *   `thread.cpp`: 執行緒管理 (Thread Pool)。

*   **其他**
    *   `function_list.cpp`: 內部函式列表管理。
    *   `invert.cpp`: 矩陣反轉 (用於 MCT)。
    *   `ht_dec.cpp`: High Throughput (HTJ2K) 解碼支援。

